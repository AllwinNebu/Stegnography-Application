import os
import customtkinter as ctk
from tkinter import filedialog, messagebox
from Crypto.Cipher import AES, Blowfish
from Crypto.Util.Padding import pad, unpad
from PIL import Image
import numpy as np
from pathlib import Path
import struct
import wave
import soundfile as sf
import math
import threading
import time
import scipy.signal
import concurrent.futures
from io import BytesIO
import mmap
import asyncio
import aiofiles
import cv2
import psutil

# Initialize GPU support flags
HAS_GPU = False
HAS_CUDA = False

# Enhanced CUDA kernels for better performance
CUDA_KERNEL = """
// Optimized pixel processing kernel with shared memory
__global__ void process_pixels(unsigned char *pixels, const unsigned char *data_bits, 
                             int n, int channels) {
    __shared__ unsigned char shared_data[1024];
    int tid = threadIdx.x;
    int bid = blockIdx.x;
    int idx = bid * blockDim.x + tid;
    
    // Load data into shared memory
    if (idx < n && tid < 1024) {
        shared_data[tid] = data_bits[idx];
    }
    __syncthreads();
    
    // Process pixels with coalesced memory access
    if (idx < n) {
        int pixel_idx = idx * channels;
        for (int c = 0; c < channels; c++) {
            pixels[pixel_idx + c] = (pixels[pixel_idx + c] & 0xFE) | 
                                  (shared_data[tid] >> c & 1);
        }
    }
}

// Optimized bit extraction kernel with shared memory
__global__ void extract_bits(const unsigned char *pixels, unsigned char *data_bits, 
                           int n, int channels) {
    __shared__ unsigned char shared_pixels[1024];
    int tid = threadIdx.x;
    int bid = blockIdx.x;
    int idx = bid * blockDim.x + tid;
    
    // Load pixels into shared memory
    if (idx < n && tid < 1024) {
        int pixel_idx = idx * channels;
        shared_pixels[tid] = 0;
        for (int c = 0; c < channels; c++) {
            shared_pixels[tid] |= (pixels[pixel_idx + c] & 1) << c;
        }
    }
    __syncthreads();
    
    // Extract bits with coalesced memory access
    if (idx < n) {
        data_bits[idx] = shared_pixels[tid];
    }
}

// Add encryption/decryption kernel for GPU acceleration
__global__ void crypto_kernel(unsigned char *data, const unsigned char *key, 
                            int n, int block_size) {
    __shared__ unsigned char shared_key[32];
    int tid = threadIdx.x;
    int bid = blockIdx.x;
    int idx = bid * blockDim.x + tid;
    
    // Load key into shared memory
    if (tid < 32) {
        shared_key[tid] = key[tid];
    }
    __syncthreads();
    
    // Process data blocks
    if (idx < n) {
        int block_idx = idx * block_size;
        for (int i = 0; i < block_size; i++) {
            data[block_idx + i] ^= shared_key[i % 32];
        }
    }
}
"""

try:
    import cupy as cp
    HAS_GPU = True
except ImportError:
    HAS_GPU = False

if HAS_GPU:
    try:
        import pycuda.autoinit
        import pycuda.driver as cuda
        from pycuda.compiler import SourceModule
        HAS_CUDA = True
        CUDA_MODULE = SourceModule(CUDA_KERNEL)
        PROCESS_PIXELS_KERNEL = CUDA_MODULE.get_function("process_pixels")
        EXTRACT_BITS_KERNEL = CUDA_MODULE.get_function("extract_bits")
        CRYPTO_KERNEL = CUDA_MODULE.get_function("crypto_kernel")
        
        # Configure optimal thread block sizes
        BLOCK_SIZE = 256
        MAX_GRID_SIZE = 65535
        
        # Initialize CUDA stream for async operations
        CUDA_STREAM = cuda.Stream()
    except (ImportError, Exception) as e:
        print(f"CUDA initialization failed: {str(e)}")
        HAS_CUDA = False
        HAS_GPU = False

class SteganographyApp:
    def __init__(self):
        # Make GPU flags accessible
        global HAS_GPU, HAS_CUDA
        
        # Define block sizes
        self.AES_BLOCK_SIZE = 16
        self.BLOWFISH_BLOCK_SIZE = 8
        
        # Initialize GPU context if available
        self.gpu_context = None
        self.pixel_kernel = None
        
        if HAS_GPU and HAS_CUDA:
            try:
                self.gpu_context = cp.cuda.Device(0).use()
                # CUDA kernel for pixel processing
                self.pixel_kernel = PROCESS_PIXELS_KERNEL
            except Exception as e:
                print(f"GPU initialization failed: {str(e)}")
                HAS_GPU = False
                HAS_CUDA = False
                self.gpu_context = None
                self.pixel_kernel = None
        
        # Optimize thread pool size based on system resources
        self.cpu_count = psutil.cpu_count(logical=False) or 4  # Physical cores, fallback to 4
        self.memory_gb = psutil.virtual_memory().total / (1024**3)
        self.max_workers = min(32, self.cpu_count * 4)
        
        # Define color scheme
        self.colors = {
            "primary": "#2D5AF0",  # Modern blue
            "primary_hover": "#1E3FA8",
            "secondary": "#404040",  # Dark gray
            "accent": "#6C63FF",  # Purple accent
            "success": "#28A745",  # Green
            "error": "#DC3545",  # Red
            "background": "#1A1A1A",  # Dark background
            "surface": "#2D2D2D",  # Slightly lighter surface
            "text": "#FFFFFF"  # White text
        }
        
        self.setup_gui()
        self.processing = False
        
    def setup_gui(self):
        # Configure appearance
        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("blue")
        
        # Configure main window
        self.window = ctk.CTk()
        self.window.title("Steganography Pro")
        self.window.geometry("800x600")  # Reduced from 1000x700
        self.window.configure(fg_color=self.colors["background"])
        
        # Create header
        header_frame = ctk.CTkFrame(self.window, fg_color="transparent")
        header_frame.pack(padx=20, pady=(20,10), fill="x")
        
        title_label = ctk.CTkLabel(
            header_frame, 
            text="Steganography Pro",
            font=ctk.CTkFont(size=24, weight="bold"),
            text_color=self.colors["text"]
        )
        title_label.pack(side="left")
        
        # Create main tabview with custom styling
        self.tabview = ctk.CTkTabview(
            self.window,
            fg_color=self.colors["surface"],
            segmented_button_fg_color=self.colors["secondary"],
            segmented_button_selected_color=self.colors["primary"],
            segmented_button_selected_hover_color=self.colors["primary_hover"]
        )
        self.tabview.pack(padx=20, pady=10, fill="both", expand=True)
        
        self.tab_hide = self.tabview.add("Hide Data")
        self.tab_extract = self.tabview.add("Extract Data")
        
        self.setup_hide_tab()
        self.setup_extract_tab()
        
    def setup_hide_tab(self):
        # Create main container frame with grid layout
        self.main_container = ctk.CTkFrame(self.tab_hide, fg_color="transparent")
        self.main_container.pack(fill="both", expand=True, padx=20, pady=20)
        self.main_container.grid_rowconfigure(0, weight=1)
        self.main_container.grid_columnconfigure(0, weight=1)
        
        # Create scrollable frame with custom styling
        self.content_frame = ctk.CTkScrollableFrame(
            self.main_container,
            fg_color=self.colors["surface"],
            corner_radius=10
        )
        self.content_frame.grid(row=0, column=0, sticky="nsew", padx=5, pady=5)
        
        # Carrier image frame with modern styling
        file_frame = ctk.CTkFrame(self.content_frame, fg_color="transparent")
        file_frame.pack(padx=20, pady=15, fill="x")
        
        ctk.CTkLabel(
            file_frame,
            text="Carrier Image",
            font=ctk.CTkFont(size=14, weight="bold"),
            text_color=self.colors["text"]
        ).pack(anchor="w")
        
        input_frame = ctk.CTkFrame(file_frame, fg_color="transparent")
        input_frame.pack(fill="x", pady=(5,0))
        
        self.image_path = ctk.CTkEntry(
            input_frame,
            width=400,
            height=35,
            placeholder_text="Select an image file...",
            fg_color=self.colors["secondary"]
        )
        self.image_path.pack(side="left", padx=(0,10), fill="x", expand=True)
        
        ctk.CTkButton(
            input_frame,
            text="Browse",
            command=self.browse_image,
            width=100,
            height=35,
            fg_color=self.colors["primary"],
            hover_color=self.colors["primary_hover"]
        ).pack(side="right")
        
        # Data type selection with modern styling
        type_frame = ctk.CTkFrame(self.content_frame, fg_color="transparent")
        type_frame.pack(padx=20, pady=15, fill="x")
        
        ctk.CTkLabel(
            type_frame,
            text="Data Type",
            font=ctk.CTkFont(size=14, weight="bold"),
            text_color=self.colors["text"]
        ).pack(anchor="w")
        
        self.data_type = ctk.CTkComboBox(
            type_frame,
            values=["Text", "Audio", "Image"],
            height=35,
            fg_color=self.colors["secondary"],
            button_color=self.colors["primary"],
            button_hover_color=self.colors["primary_hover"],
            dropdown_fg_color=self.colors["surface"]
        )
        self.data_type.pack(fill="x", padx=0, pady=(5,0))
        self.data_type.set("Text")
        self.data_type.configure(command=self.on_data_type_change)
        
        # Input frames container
        self.input_container = ctk.CTkFrame(self.content_frame, fg_color="transparent")
        self.input_container.pack(padx=20, pady=15, fill="x")
        
        # Text input frame with modern styling
        self.text_frame = ctk.CTkFrame(self.input_container, fg_color="transparent")
        ctk.CTkLabel(
            self.text_frame,
            text="Text to Hide",
            font=ctk.CTkFont(size=14, weight="bold"),
            text_color=self.colors["text"]
        ).pack(anchor="w")
        self.text_input = ctk.CTkTextbox(
            self.text_frame,
            height=150,
            fg_color=self.colors["secondary"],
            corner_radius=8
        )
        self.text_input.pack(fill="x", padx=0, pady=(5,0))
        
        # Audio input frame
        self.audio_frame = ctk.CTkFrame(self.input_container, fg_color="transparent")
        ctk.CTkLabel(
            self.audio_frame,
            text="Audio File",
            font=ctk.CTkFont(size=14, weight="bold"),
            text_color=self.colors["text"]
        ).pack(anchor="w")
        
        audio_input_frame = ctk.CTkFrame(self.audio_frame, fg_color="transparent")
        audio_input_frame.pack(fill="x", pady=(5,0))
        
        self.audio_path = ctk.CTkEntry(
            audio_input_frame,
            placeholder_text="Select an audio file...",
            height=35,
            fg_color=self.colors["secondary"]
        )
        self.audio_path.pack(side="left", padx=(0,10), fill="x", expand=True)
        
        self.audio_browse_btn = ctk.CTkButton(
            audio_input_frame,
            text="Browse",
            command=self.browse_audio,
            width=100,
            height=35,
            fg_color=self.colors["primary"],
            hover_color=self.colors["primary_hover"]
        )
        self.audio_browse_btn.pack(side="right")
        
        # Image input frame
        self.image_input_frame = ctk.CTkFrame(self.input_container, fg_color="transparent")
        ctk.CTkLabel(
            self.image_input_frame,
            text="Image to Hide",
            font=ctk.CTkFont(size=14, weight="bold"),
            text_color=self.colors["text"]
        ).pack(anchor="w")
        
        image_input_container = ctk.CTkFrame(self.image_input_frame, fg_color="transparent")
        image_input_container.pack(fill="x", pady=(5,0))
        
        self.hidden_image_path = ctk.CTkEntry(
            image_input_container,
            placeholder_text="Select an image to hide...",
            height=35,
            fg_color=self.colors["secondary"]
        )
        self.hidden_image_path.pack(side="left", padx=(0,10), fill="x", expand=True)
        
        ctk.CTkButton(
            image_input_container,
            text="Browse",
            command=self.browse_hidden_image,
            width=100,
            height=35,
            fg_color=self.colors["primary"],
            hover_color=self.colors["primary_hover"]
        ).pack(side="right")
        
        # Show text frame by default
        self.text_frame.pack(fill="x", padx=0, pady=0)
        
        # Key frame with modern styling
        key_frame = ctk.CTkFrame(self.content_frame, fg_color="transparent")
        key_frame.pack(padx=20, pady=15, fill="x")
        
        ctk.CTkLabel(
            key_frame,
            text="Encryption Key",
            font=ctk.CTkFont(size=14, weight="bold"),
            text_color=self.colors["text"]
        ).pack(anchor="w")
        
        self.key_input = ctk.CTkEntry(
            key_frame,
            show="•",
            height=35,
            placeholder_text="Enter your encryption key...",
            fg_color=self.colors["secondary"]
        )
        self.key_input.pack(fill="x", padx=0, pady=(5,0))
        
        # Progress frame with modern styling
        self.progress_frame = ctk.CTkFrame(self.content_frame, fg_color="transparent")
        self.progress_frame.pack(padx=20, pady=15, fill="x")
        
        self.progress_bar = ctk.CTkProgressBar(
            self.progress_frame,
            fg_color=self.colors["secondary"],
            progress_color=self.colors["primary"]
        )
        self.progress_bar.pack(fill="x", padx=0)
        self.progress_bar.set(0)
        self.progress_bar.pack_forget()
        
        self.status_label = ctk.CTkLabel(
            self.progress_frame,
            text="",
            font=ctk.CTkFont(size=12),
            text_color=self.colors["text"]
        )
        self.status_label.pack()
        self.status_label.pack_forget()
        
        # Button frame with modern styling
        self.button_frame = ctk.CTkFrame(self.main_container, fg_color="transparent", height=60)
        self.button_frame.grid(row=1, column=0, sticky="ew", pady=(10,0))
        self.button_frame.grid_columnconfigure(0, weight=1)
        self.button_frame.grid_propagate(False)
        
        self.hide_button = ctk.CTkButton(
            self.button_frame,
            text="Hide Data",
            command=self.start_hide_data,
            width=200,
            height=45,
            fg_color=self.colors["primary"],
            hover_color=self.colors["primary_hover"],
            font=ctk.CTkFont(size=14, weight="bold")
        )
        self.hide_button.grid(row=0, column=0, pady=10)
        
    def setup_extract_tab(self):
        # Main container for extract tab
        main_frame = ctk.CTkFrame(self.tab_extract, fg_color="transparent")
        main_frame.pack(fill="both", expand=True, padx=20, pady=20)
        
        # File selection frame
        file_frame = ctk.CTkFrame(main_frame, fg_color=self.colors["surface"], corner_radius=10)
        file_frame.pack(padx=0, pady=(0,20), fill="x")
        
        ctk.CTkLabel(
            file_frame,
            text="Stego Image",
            font=ctk.CTkFont(size=14, weight="bold"),
            text_color=self.colors["text"]
        ).pack(anchor="w", padx=20, pady=(15,5))
        
        input_frame = ctk.CTkFrame(file_frame, fg_color="transparent")
        input_frame.pack(fill="x", padx=20, pady=(0,15))
        
        self.extract_image_path = ctk.CTkEntry(
            input_frame,
            placeholder_text="Select an image to extract data from...",
            height=35,
            fg_color=self.colors["secondary"]
        )
        self.extract_image_path.pack(side="left", padx=(0,10), fill="x", expand=True)
        
        ctk.CTkButton(
            input_frame,
            text="Browse",
            command=self.browse_extract_image,
            width=100,
            height=35,
            fg_color=self.colors["primary"],
            hover_color=self.colors["primary_hover"]
        ).pack(side="right")
        
        # Key frame
        key_frame = ctk.CTkFrame(main_frame, fg_color=self.colors["surface"], corner_radius=10)
        key_frame.pack(padx=0, pady=(0,20), fill="x")
        
        ctk.CTkLabel(
            key_frame,
            text="Decryption Key",
            font=ctk.CTkFont(size=14, weight="bold"),
            text_color=self.colors["text"]
        ).pack(anchor="w", padx=20, pady=(15,5))
        
        self.extract_key_input = ctk.CTkEntry(
            key_frame,
            show="•",
            height=35,
            placeholder_text="Enter your decryption key...",
            fg_color=self.colors["secondary"]
        )
        self.extract_key_input.pack(fill="x", padx=20, pady=(0,15))
        
        # Data type indicator
        type_frame = ctk.CTkFrame(main_frame, fg_color=self.colors["surface"], corner_radius=10)
        type_frame.pack(padx=0, pady=(0,20), fill="x")
        
        self.extracted_type_label = ctk.CTkLabel(
            type_frame,
            text="",
            font=ctk.CTkFont(size=14),
            text_color=self.colors["text"]
        )
        self.extracted_type_label.pack(padx=20, pady=15)
        
        # Extract button
        self.extract_button = ctk.CTkButton(
            main_frame,
            text="Extract Data",
            command=self.extract_data,
            width=200,
            height=45,
            fg_color=self.colors["primary"],
            hover_color=self.colors["primary_hover"],
            font=ctk.CTkFont(size=14, weight="bold")
        )
        self.extract_button.pack(pady=(0,20))
        
        # Output text area
        self.output_text = ctk.CTkTextbox(
            main_frame,
            height=200,
            fg_color=self.colors["surface"],
            corner_radius=10,
            font=ctk.CTkFont(size=12)
        )
        self.output_text.pack(padx=0, pady=0, fill="both", expand=True)
    
    def on_data_type_change(self, choice):
        # Hide all frames
        for frame in [self.text_frame, self.audio_frame, self.image_input_frame]:
            frame.pack_forget()
        
        # Show selected frame
        if choice == "Text":
            self.text_frame.pack(fill="x", padx=5, pady=5)
        elif choice == "Audio":
            self.audio_frame.pack(fill="x", padx=5, pady=5)
        elif choice == "Image":
            self.image_input_frame.pack(fill="x", padx=5, pady=5)
        
        # Force update
        self.window.update_idletasks()
    
    def browse_image(self):
        try:
            file_path = filedialog.askopenfilename(
                title="Select Carrier Image",
                filetypes=[("Image files", "*.png *.jpg *.jpeg")]
            )
            if file_path:
                # Show progress indicators
                self.progress_bar.pack(fill="x", padx=5)
                self.status_label.pack()
                self.update_progress(0, "Processing image...")
                
                # Disable the hide button during processing
                self.hide_button.configure(state="disabled")
                
                # Start processing thread
                thread = threading.Thread(target=self.process_image_thread, args=(file_path,))
                thread.daemon = True
                thread.start()
        except Exception as e:
            self.show_error(f"Error selecting image: {str(e)}")
            self.reset_ui_state()

    def process_image_thread(self, file_path):
        try:
            self.update_progress(0.1, "Verifying image...")
            
            # Load image using OpenCV for better performance
            img_array = cv2.imread(file_path, cv2.IMREAD_COLOR)
            if img_array is None:
                raise ValueError("Failed to load image")
            
            # Convert to RGB
            img_array = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)
            
            # Convert to PIL Image
            image = Image.fromarray(img_array)
            
            # Check if resizing is needed
            file_size = os.path.getsize(file_path)
            should_enlarge = file_size < 5 * 1024 * 1024  # 5MB threshold
            
            if should_enlarge:
                # Calculate new dimensions
                current_width, current_height = image.size
                scale_factor = min(2.0, 5 * 1024 * 1024 / (current_width * current_height * 3))
                new_width = max(1, int(round(current_width * scale_factor)))
                new_height = max(1, int(round(current_height * scale_factor)))
                
                self.update_progress(0.4, "Optimizing image...")
                
                # Use OpenCV for faster resizing
                img_array = cv2.resize(
                    img_array,
                    (new_width, new_height),
                    interpolation=cv2.INTER_LINEAR
                )
                
                # Save as PNG to preserve quality
                output_path = str(Path(file_path).parent / f"enlarged_{Path(file_path).stem}.png")
                
                # Save with minimal compression
                cv2.imwrite(
                    output_path,
                    cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR),
                    [cv2.IMWRITE_PNG_COMPRESSION, 0]
                )
                
                # Update UI
                self.window.after(0, lambda: self.finish_image_processing(
                    output_path,
                    current_width,
                    current_height,
                    new_width,
                    new_height
                ))
            else:
                # For larger images, use original
                self.window.after(0, lambda: self.finish_image_processing(
                    file_path,
                    image.size[0],
                    image.size[1],
                    image.size[0],
                    image.size[1]
                ))
                
        except Exception as e:
            self.window.after(0, lambda: self.show_error(
                "Failed to process the image.\n\n"
                f"Error details: {str(e)}\n\n"
                "Please try:\n"
                "1. Using a different image\n"
                "2. Using a PNG format image\n"
                "3. Ensuring you have enough disk space"
            ))
        finally:
            self.window.after(0, self.reset_ui_state)

    def finish_image_processing(self, file_path, current_width, current_height, new_width, new_height):
        try:
            # Update the image path
            self.image_path.delete(0, "end")
            self.image_path.insert(0, file_path)
            
            # Update status label instead of showing popup
            self.status_label.configure(text=f"Image ready: {new_width}x{new_height}")
            
        except Exception as e:
            self.show_error(f"Error updating UI: {str(e)}")
    
    def browse_audio(self):
        try:
            file_path = filedialog.askopenfilename(
                title="Select Audio File",
                filetypes=[("Audio files", "*.wav *.mp3")],
                initialdir=os.path.expanduser("~")
            )
            
            if file_path:
                if file_path.lower().endswith('.mp3'):
                    try:
                        self.status_label.configure(text="Converting MP3 to WAV...")
                        self.status_label.pack()
                        self.window.update()
                        
                        # Read MP3 and convert to WAV
                        data, samplerate = sf.read(file_path)
                        wav_path = str(Path(file_path).with_suffix('.wav'))
                        sf.write(wav_path, data, samplerate)
                        
                        self.status_label.configure(text="Audio ready")
                        file_path = wav_path
                    except Exception as e:
                        self.status_label.pack_forget()
                        messagebox.showerror("Error", f"Failed to convert audio file: {str(e)}")
                        return
                
                self.audio_path.delete(0, "end")
                self.audio_path.insert(0, file_path)
                
        except Exception as e:
            messagebox.showerror("Error", 
                "Failed to process the audio file.\n\n"
                f"Error details: {str(e)}\n\n"
                "Please try:\n"
                "1. Using a different audio file\n"
                "2. Checking if the file is corrupted\n"
                "3. Using a WAV file instead of MP3")
    
    def browse_hidden_image(self):
        file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.png *.jpg *.jpeg")])
        if file_path:
            self.hidden_image_path.delete(0, "end")
            self.hidden_image_path.insert(0, file_path)
    
    def browse_extract_image(self):
        try:
            file_path = filedialog.askopenfilename(
                title="Select Stego Image",
                filetypes=[("PNG Files", "*.png")],
                initialdir=os.path.expanduser("~")
            )
            
            if file_path:
                try:
                    # Verify the image can be opened
                    with Image.open(file_path) as img:
                        img.verify()
                    
                    # Update the UI
                    self.extract_image_path.delete(0, "end")
                    self.extract_image_path.insert(0, file_path)
                    
                    # Clear previous results
                    self.extracted_type_label.configure(text="")
                    self.output_text.delete("1.0", "end")
                    
                except Exception as e:
                    self.show_error(
                        "Failed to verify the selected image.\n\n"
                        f"Error details: {str(e)}\n\n"
                        "Please ensure the file is a valid PNG image file."
                    )
                    
        except Exception as e:
            self.show_error(
                "Failed to open file dialog.\n\n"
                f"Error details: {str(e)}"
            )
    
    def start_hide_data(self):
        if self.processing:
            return
            
        self.processing = True
        # Disable button but keep it visible
        self.hide_button.configure(state="disabled", text="Processing...")
        # Show progress indicators
        self.progress_bar.pack(fill="x", padx=5)
        self.status_label.pack()
        self.update_progress(0, "Preparing...")
        
        # Start processing thread
        thread = threading.Thread(target=self.hide_data_thread)
        thread.daemon = True
        thread.start()

    def hide_data_thread(self):
        try:
            image_path = self.image_path.get()
            key = self.key_input.get()
            data_type = self.data_type.get()
            
            if not image_path:
                self.show_error("Please select a carrier image first")
                return
            if not key:
                self.show_error("Please enter an encryption key")
                return
            
            try:
                self.update_progress(0.1, "Reading input data...")
                
                # Get data based on type
                if data_type == "Text":
                    data = self.text_input.get("1.0", "end-1c")
                    if not data:
                        raise ValueError("Please enter text to hide")
                    data = data.encode('utf-8')
                elif data_type == "Audio":
                    audio_path = self.audio_path.get()
                    if not audio_path:
                        raise ValueError("Please select an audio file")
                    try:
                        self.update_progress(0.2, "Processing audio file...")
                        # Read audio data with original format preserved
                        data = self.process_audio_file(audio_path)
                        self.status_label.configure(text="Audio file processed")
                    except Exception as e:
                        raise ValueError(f"Failed to process audio file: {str(e)}")
                elif data_type == "Image":
                    hidden_image_path = self.hidden_image_path.get()
                    if not hidden_image_path:
                        raise ValueError("Please select an image to hide")
                    try:
                        self.update_progress(0.2, "Processing image to hide...")
                        data = self.load_image_to_data(hidden_image_path)
                    except Exception as e:
                        raise ValueError(f"Failed to process image to hide: {str(e)}")
                
                self.update_progress(0.3, "Processing carrier image...")
                try:
                    carrier_image = Image.open(image_path)
                    carrier_image = self.resize_image_to_fit_data(carrier_image, data)
                except Exception as e:
                    raise ValueError(f"Failed to process carrier image: {str(e)}")
                
                self.update_progress(0.5, "Encrypting data...")
                # Prepare output path
                output_path = str(Path(image_path).parent / f"stego_{Path(image_path).name}")
                
                # Hide the data with progress updates
                self.hide_data_in_image_with_progress(carrier_image, data, key, output_path, data_type)
                
                self.update_progress(1.0, "Complete!")
                
                # Show success message
                msg = (
                    f"Data hidden successfully!\n\n"
                    f"Output image: {output_path}\n"
                    f"Image size: {carrier_image.size[0]}x{carrier_image.size[1]}\n"
                    f"Data type: {data_type}"
                )
                self.show_success("Success", msg)
                
            except ValueError as e:
                self.show_error(str(e))
            except Exception as e:
                self.show_error(
                    "An unexpected error occurred.\n\n"
                    f"Error details: {str(e)}\n\n"
                    "Please try:\n"
                    "1. Using a smaller image or data size\n"
                    "2. Using different input files\n"
                    "3. Checking if you have enough disk space"
                )
                
        finally:
            # Reset UI state
            self.window.after(0, self.reset_ui_state)

    def hide_data_in_image_with_progress(self, image, data, key, output_path, data_type):
        """Optimized data hiding with direct bit manipulation"""
        try:
            self.update_progress(0.5, "Encrypting data...")
            key = self.generate_key(key)
            encrypted_data = self.process_large_data(data, key, encrypt=True)
            
            self.update_progress(0.6, "Preparing data...")
            
            # Prepare header with fixed sizes
            type_prefix = b'TXT' if data_type == "Text" else b'AUD' if data_type == "Audio" else b'IMG'
            length_prefix = struct.pack('>Q', len(encrypted_data))
            data_to_hide = type_prefix + length_prefix + encrypted_data
            
            # Calculate required capacity
            required_bits = len(data_to_hide) * 8
            
            # Convert image to numpy array
            image_array = np.array(image, dtype=np.uint8)
            available_bits = image_array.size
            
            if required_bits > available_bits:
                raise ValueError(
                    f"Image capacity ({available_bits // 8} bytes) is insufficient for data size ({len(data_to_hide)} bytes)"
                )
            
            self.update_progress(0.7, "Embedding data...")
            
            # Convert data to bits
            data_bits = np.unpackbits(np.frombuffer(data_to_hide, dtype=np.uint8))
            
            # Create a copy of the flattened array
            modified_pixels = image_array.ravel().copy()
            
            # Embed data bits
            modified_pixels[:required_bits] = (modified_pixels[:required_bits] & 0xFE) | data_bits
            
            # Reshape back to original dimensions
            modified_array = modified_pixels.reshape(image_array.shape)
            
            self.update_progress(0.9, "Saving image...")
            
            # Ensure output path has .png extension
            output_path = str(Path(output_path).with_suffix('.png'))
            
            # Convert to BGR for OpenCV
            img_bgr = cv2.cvtColor(modified_array, cv2.COLOR_RGB2BGR)
            
            # Save with PNG compression settings
            success = cv2.imwrite(output_path, img_bgr, [cv2.IMWRITE_PNG_COMPRESSION, 0])
            
            if not success:
                raise ValueError("Failed to save the image")
            
            self.update_progress(1.0, "Complete!")
            
        except Exception as e:
            raise ValueError(f"Error during data hiding: {str(e)}")

    def _process_on_cpu(self, image_array, data_bits, required_bits):
        """Optimized CPU processing with vectorized operations"""
        # Calculate optimal chunk size based on available memory
        chunk_size = min(self.chunk_size, required_bits)
        
        # Process in parallel using thread pool
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = []
            
            for i in range(0, required_bits, chunk_size):
                end = min(i + chunk_size, required_bits)
                chunk_data = (
                    image_array.ravel()[i:end],
                    data_bits[i:end]
                )
                futures.append(executor.submit(self.process_pixel_chunk, chunk_data))
            
            # Collect results
            for i, future in enumerate(futures):
                start = i * chunk_size
                end = min(start + chunk_size, required_bits)
                image_array.ravel()[start:end] = future.result()

    def process_pixel_chunk(self, chunk_data):
        """Process a chunk of pixels on CPU"""
        pixels, data_bits = chunk_data
        pixels = pixels.copy()
        pixels[:len(data_bits)] = (pixels[:len(data_bits)] & 0xFE) | data_bits
        return pixels

    def reset_ui_state(self):
        try:
            # Re-enable and reset button text
            self.hide_button.configure(state="normal", text="Hide Data")
            # Hide progress indicators
            self.progress_bar.pack_forget()
            self.status_label.pack_forget()
            self.processing = False
            # Force update to ensure button is visible
            self.button_frame.update()
        except Exception as e:
            print(f"Error resetting UI state: {str(e)}")
            # Fallback reset
            self.processing = False
            if hasattr(self, 'hide_button'):
                self.hide_button.configure(state="normal", text="Hide Data")

    def show_error(self, message):
        """Display error message with improved formatting"""
        self.window.after(0, lambda: messagebox.showerror(
            "Error",
            message,
            icon=messagebox.ERROR,
            parent=self.window
        ))

    def show_success(self, title, message):
        """Display success message with improved formatting"""
        self.window.after(0, lambda: messagebox.showinfo(
            title,
            message,
            icon=messagebox.INFO,
            parent=self.window
        ))

    def update_progress(self, value, status=""):
        self.progress_bar.set(value)
        self.status_label.configure(text=status)
        self.window.update()

    def resize_image_to_fit_data(self, image, data):
        try:
            # Calculate required bits for data
            required_bits = len(data) * 8
            current_width, current_height = image.size
            current_capacity = current_width * current_height * 3  # 3 channels (RGB)
            
            # Set maximum dimensions and compression thresholds
            MAX_WIDTH = 16000
            MAX_HEIGHT = 16000
            MAX_TOTAL_PIXELS = MAX_WIDTH * MAX_HEIGHT
            COMPRESSION_THRESHOLD = 20 * 1024 * 1024  # 20MB
            
            # Calculate required pixels and check if data is too large
            required_pixels = math.ceil(required_bits / 3)  # Each pixel can store 3 bits (RGB)
            if required_pixels > MAX_TOTAL_PIXELS:
                raise ValueError(
                    f"Data size ({len(data)} bytes) is too large for the maximum allowed image size.\n"
                    f"Maximum capacity: {(MAX_TOTAL_PIXELS * 3) // 8} bytes\n"
                    f"Required capacity: {len(data)} bytes\n"
                    "Please reduce the data size or split it into smaller chunks."
                )
            
            # If current image is large enough, use it as is
            if current_capacity >= required_bits:
                return image
            
            # Calculate new dimensions while maintaining aspect ratio
            aspect_ratio = current_width / current_height
            new_pixels = required_pixels
            
            if aspect_ratio > 1:
                # Wider than tall
                new_width = min(MAX_WIDTH, math.ceil(math.sqrt(new_pixels * aspect_ratio)))
                new_height = min(MAX_HEIGHT, math.ceil(new_width / aspect_ratio))
            else:
                # Taller than wide or square
                new_height = min(MAX_HEIGHT, math.ceil(math.sqrt(new_pixels / aspect_ratio)))
                new_width = min(MAX_WIDTH, math.ceil(new_height * aspect_ratio))
            
            # Ensure minimum dimensions
            new_width = max(1, new_width)
            new_height = max(1, new_height)
            
            # Verify final capacity
            final_capacity = new_width * new_height * 3
            if required_bits > final_capacity:
                # If still not enough, try maximum dimensions
                new_width = MAX_WIDTH
                new_height = MAX_HEIGHT
                final_capacity = new_width * new_height * 3
            
            if required_bits > final_capacity:
                raise ValueError(
                    f"Data size ({len(data)} bytes) is too large for the maximum allowed image size.\n"
                    f"Maximum capacity: {final_capacity // 8} bytes\n"
                    f"Required capacity: {len(data)} bytes\n"
                    "Please reduce the data size or split it into smaller chunks."
                )
            
            # Use faster resampling method for large images
            resampling_method = Image.Resampling.BILINEAR if max(new_width, new_height) > 1000 else Image.Resampling.LANCZOS
            
            # Resize image with optimized settings
            resized_image = image.resize(
                (new_width, new_height),
                resampling_method,
                reducing_gap=3.0
            )
            
            return resized_image
            
        except Exception as e:
            raise ValueError(f"Error resizing image: {str(e)}")

    def load_image_to_data(self, image_path):
        # Convert image to binary data to be hidden
        with Image.open(image_path) as image:
            # Convert to RGB mode
            image = image.convert('RGB')
            # Get dimensions
            width, height = image.size
            # Pack dimensions at the start
            dimensions = struct.pack('>II', width, height)
            # Get image data
            image_data = image.tobytes()
            return dimensions + image_data

    def extract_data(self):
        try:
            image_path = self.extract_image_path.get()
            key = self.extract_key_input.get()
            
            if not image_path:
                self.show_error("Please select an image to extract data from.")
                return
            if not key:
                self.show_error("Please enter the decryption key.")
                return
            
            # Show processing message
            self.extract_button.configure(state="disabled", text="Processing...")
            self.window.update()
            
            try:
                data, data_type = self.extract_data_from_image(image_path, key)
                
                self.extracted_type_label.configure(text=f"Extracted Data Type: {data_type}")
                
                if data_type == "Text":
                    try:
                        text = data.decode('utf-8')
                        self.output_text.delete("1.0", "end")
                        self.output_text.insert("1.0", text)
                        self.show_success("Success", "Text extracted successfully!")
                    except Exception as e:
                        raise ValueError("Failed to decode the extracted text. The key may be incorrect.")
                elif data_type == "Audio":
                    try:
                        self.extract_audio_data(data)
                        self.show_success("Success", "Audio extracted successfully!")
                    except Exception as e:
                        raise ValueError("Failed to process the extracted audio. The key may be incorrect.")
                else:
                    try:
                        self.extract_image_data(data)
                        self.show_success("Success", "Image extracted successfully!")
                    except Exception as e:
                        raise ValueError("Failed to process the extracted image. The key may be incorrect.")
                    
            except ValueError as e:
                self.show_error(str(e))
            except Exception as e:
                self.show_error(
                    "Failed to extract data from the image.\n\n"
                    "Please verify:\n"
                    "1. You entered the correct decryption key\n"
                    "2. The image contains hidden data\n"
                    "3. The image file is not corrupted"
                )
            finally:
                # Reset button state
                self.extract_button.configure(state="normal", text="Extract Data")
                self.window.update()
                
        except Exception as e:
            self.show_error(str(e))
            self.extract_button.configure(state="normal", text="Extract Data")
            self.window.update()

    def process_audio_file(self, audio_path):
        """Optimized method to process audio files with buffered reading"""
        try:
            with wave.open(audio_path, 'rb') as wav_file:
                # Get all parameters
                params = wav_file.getparams()
                
                # Pack parameters with fixed-size format
                params_data = struct.pack('>6I', 
                    params.nchannels, params.sampwidth, params.framerate,
                    params.nframes, 4, 4  # Fixed size for comptype and compname
                )
                # Add fixed-size compression info
                params_data += b'NONE'.ljust(4)  # comptype
                params_data += b'NONE'.ljust(4)  # compname
                
                # Read audio data in chunks for better memory efficiency
                CHUNK_SIZE = 1024 * 1024  # 1MB chunks
                audio_chunks = []
                while True:
                    chunk = wav_file.readframes(CHUNK_SIZE)
                    if not chunk:
                        break
                    audio_chunks.append(chunk)
                
                # Combine all data
                audio_data = b''.join(audio_chunks)
                
                # Add a marker for validation
                marker = b'WAVMARK'
                full_data = marker + params_data + audio_data
                
                return full_data
        except Exception as e:
            raise ValueError(f"Failed to process audio file: {str(e)}")

    def extract_audio_data(self, data):
        """Optimized method to extract and save audio data with buffered writing"""
        try:
            # Check for marker
            marker_size = 7  # len(b'WAVMARK')
            if not data.startswith(b'WAVMARK'):
                raise ValueError("Invalid audio data format")
            
            # Extract parameters (after marker)
            header_size = struct.calcsize('>6I')
            params = struct.unpack('>6I', data[marker_size:marker_size + header_size])
            nchannels, sampwidth, framerate, nframes, _, _ = params
            
            # Skip fixed-size compression info
            pos = marker_size + header_size + 8  # 8 bytes for compression info
            
            # Get audio data
            audio_data = data[pos:]
            
            # Validate basic audio parameters
            if not (1 <= nchannels <= 8 and 1 <= sampwidth <= 4 and 
                    8000 <= framerate <= 192000 and nframes > 0):
                raise ValueError("Invalid audio parameters")
            
            # Save audio file with buffered writing
            output_path = str(Path(self.extract_image_path.get()).parent / "extracted_audio.wav")
            with wave.open(output_path, 'wb') as wav_file:
                wav_file.setnchannels(nchannels)
                wav_file.setsampwidth(sampwidth)
                wav_file.setframerate(framerate)
                wav_file.setnframes(nframes)
                wav_file.setcomptype('NONE', 'NONE')
                
                # Write in chunks for better performance
                CHUNK_SIZE = 1024 * 1024  # 1MB chunks
                for i in range(0, len(audio_data), CHUNK_SIZE):
                    chunk = audio_data[i:i + CHUNK_SIZE]
                    wav_file.writeframes(chunk)
            
            self.output_text.delete("1.0", "end")
            self.output_text.insert("1.0", 
                f"Audio extracted to: {output_path}\n\n"
                f"Audio Properties:\n"
                f"Channels: {nchannels}\n"
                f"Sample Width: {sampwidth} bytes\n"
                f"Frame Rate: {framerate} Hz\n"
                f"Frames: {nframes}"
            )
            
        except Exception as e:
            raise ValueError(f"Failed to extract audio: {str(e)}")

    def extract_image_data(self, data):
        try:
            # First 8 bytes contain the dimensions (2 integers for width and height)
            dimensions_size = struct.calcsize('>II')
            width, height = struct.unpack('>II', data[:dimensions_size])
            
            # Extract actual image data
            image_data = data[dimensions_size:]
            
            # Calculate expected data size
            expected_size = width * height * 3  # 3 bytes per pixel (RGB)
            if len(image_data) != expected_size:
                # Adjust data if needed
                if len(image_data) > expected_size:
                    image_data = image_data[:expected_size]
                else:
                    # Pad with zeros if data is too short
                    image_data = image_data + b'\0' * (expected_size - len(image_data))
            
            # Create and save the hidden image
            try:
                hidden_image = Image.frombytes('RGB', (width, height), image_data)
                output_path = str(Path(self.extract_image_path.get()).parent / "extracted_image.png")
                hidden_image.save(output_path)
                self.output_text.delete("1.0", "end")
                self.output_text.insert("1.0", f"Image extracted to: {output_path}")
            except Exception as e:
                raise ValueError(f"Failed to create image: {str(e)}")
        except Exception as e:
            raise ValueError(f"Failed to extract image: {str(e)}")

    def generate_key(self, shared_key):
        """Generate consistent keys for both AES and Blowfish"""
        try:
            # Clean and normalize the key
            shared_key = shared_key.strip()
            if not shared_key:
                raise ValueError("Empty encryption key")
            
            # Convert the shared key to bytes with consistent encoding
            key = shared_key.encode('utf-8')
            
            # Use SHA-256 to get a consistent 32-byte key
            from hashlib import sha256
            hashed = sha256(key).digest()
            
            # Use first 16 bytes for AES and last 16 bytes for Blowfish
            aes_key = hashed[:16]
            blowfish_key = hashed[16:]
            
            # Validate key lengths
            if len(aes_key) != 16 or len(blowfish_key) != 16:
                raise ValueError("Invalid key length after processing")
            
            return (aes_key, blowfish_key)
        except Exception as e:
            raise ValueError(f"Key generation failed: {str(e)}")

    def process_large_data(self, data, key, encrypt=True):
        """Process large data in chunks with proper padding"""
        try:
            if not data:
                raise ValueError("Empty data")
            
            # Ensure data length is valid for decryption
            if not encrypt and len(data) % self.AES_BLOCK_SIZE != 0:
                raise ValueError("Invalid encrypted data length")
            
            if encrypt:
                try:
                    # Add padding before encryption
                    padded_data = pad(data, self.AES_BLOCK_SIZE)
                    
                    # Process data in blocks
                    result = bytearray()
                    
                    # Process each block
                    for i in range(0, len(padded_data), self.AES_BLOCK_SIZE):
                        block = padded_data[i:i + self.AES_BLOCK_SIZE]
                        # Alternate between AES and Blowfish
                        is_aes = (i // self.AES_BLOCK_SIZE) % 2 == 0
                        encrypted_block = self.encrypt_block(block, key, is_aes)
                        result.extend(encrypted_block)
                    
                    return bytes(result)
                except Exception as e:
                    raise ValueError(f"Encryption failed: {str(e)}")
                
            else:
                try:
                    # For decryption, process block by block
                    result = bytearray()
                    
                    # Process each block
                    for i in range(0, len(data), self.AES_BLOCK_SIZE):
                        block = data[i:i + self.AES_BLOCK_SIZE]
                        # Use same alternation as encryption
                        is_aes = (i // self.AES_BLOCK_SIZE) % 2 == 0
                        decrypted_block = self.decrypt_block(block, key, is_aes)
                        result.extend(decrypted_block)
                    
                    try:
                        # Remove padding from the final result
                        if len(result) >= self.AES_BLOCK_SIZE:
                            return unpad(bytes(result), self.AES_BLOCK_SIZE)
                        else:
                            raise ValueError("Decrypted data too short")
                    except ValueError as e:
                        raise ValueError("Decryption failed. The key may be incorrect or the data is corrupted.")
                    
                except ValueError as e:
                    raise ValueError(str(e))
                except Exception as e:
                    raise ValueError(f"Decryption failed: {str(e)}")
                
        except ValueError as e:
            raise ValueError(str(e))
        except Exception as e:
            raise ValueError(f"Data processing failed: {str(e)}")

    def encrypt_block(self, block, key, is_aes):
        """Encrypt a single block using either AES or Blowfish with consistent key handling"""
        try:
            if not block or len(block) != self.AES_BLOCK_SIZE:
                raise ValueError(f"Invalid block size: {len(block) if block else 0}")
            
            if not key or len(key) != 2:
                raise ValueError("Invalid key tuple")
            
            if len(key[0]) != 16 or len(key[1]) != 16:
                raise ValueError("Invalid key lengths")
            
            if is_aes:
                cipher = AES.new(key[0], AES.MODE_ECB)
                return cipher.encrypt(block)
            else:
                cipher = Blowfish.new(key[1], Blowfish.MODE_ECB)
                return cipher.encrypt(block)
        except Exception as e:
            raise ValueError(f"Block encryption failed: {str(e)}")

    def decrypt_block(self, block, key, is_aes):
        """Decrypt a single block using either AES or Blowfish with consistent key handling"""
        try:
            if not block or len(block) != self.AES_BLOCK_SIZE:
                raise ValueError(f"Invalid block size: {len(block) if block else 0}")
            
            if not key or len(key) != 2:
                raise ValueError("Invalid key tuple")
            
            if len(key[0]) != 16 or len(key[1]) != 16:
                raise ValueError("Invalid key lengths")
            
            if is_aes:
                cipher = AES.new(key[0], AES.MODE_ECB)
                return cipher.decrypt(block)
            else:
                cipher = Blowfish.new(key[1], Blowfish.MODE_ECB)
                return cipher.decrypt(block)
        except Exception as e:
            raise ValueError(f"Block decryption failed: {str(e)}")

    def extract_data_from_image(self, image_path, key):
        """GPU-accelerated data extraction with improved key handling"""
        try:
            # Load image efficiently with OpenCV
            img_array = cv2.imread(image_path, cv2.IMREAD_COLOR)
            if img_array is None:
                raise ValueError("Failed to load image")
            
            # Convert BGR to RGB efficiently
            img_array = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)
            
            # Get dimensions and validate
            height, width, channels = img_array.shape
            total_bits = width * height * channels
            
            if total_bits < 88:  # Minimum bits needed for header
                raise ValueError("Image too small to contain hidden data")
            
            # Extract bits efficiently using numpy operations
            pixels = img_array.ravel()
            data_bits = pixels & 1
            
            # Extract type marker (first 24 bits)
            type_bits = data_bits[:24]
            type_bytes = np.packbits(type_bits).tobytes()
            
            if type_bytes not in [b'TXT', b'AUD', b'IMG']:
                raise ValueError(
                    "No valid hidden data found in this image.\n"
                    "Please verify that this is a steganographic image."
                )
            
            data_type = "Text" if type_bytes == b'TXT' else "Audio" if type_bytes == b'AUD' else "Image"
            
            # Extract length (next 64 bits)
            length_bits = data_bits[24:88]
            length = struct.unpack('>Q', np.packbits(length_bits).tobytes())[0]
            
            # Validate length
            max_possible_length = (total_bits - 88) // 8
            if length == 0 or length > max_possible_length:
                raise ValueError(
                    f"Invalid data length detected: {length} bytes\n"
                    f"Maximum possible length: {max_possible_length} bytes"
                )
            
            # Extract data bits efficiently
            required_bits = length * 8
            data_bits = data_bits[88:88 + required_bits]
            
            # Convert bits to bytes efficiently
            encrypted_data = np.packbits(data_bits).tobytes()
            
            # Generate key and decrypt data
            try:
                key_pair = self.generate_key(key)
                decrypted_data = self.process_large_data(encrypted_data, key_pair, False)
                
                # Validate decrypted data
                self._validate_extracted_data(decrypted_data, data_type)
                
                return decrypted_data, data_type
                
            except ValueError as e:
                raise ValueError(f"Decryption failed: {str(e)}")
            except Exception as e:
                raise ValueError("Decryption failed. The key may be incorrect or the data is corrupted.")
            
        except ValueError as e:
            raise ValueError(str(e))
        except Exception as e:
            raise ValueError(f"Data extraction failed: {str(e)}")

    def _validate_extracted_data(self, data, data_type):
        """Validate extracted data based on type"""
        try:
            if data_type == "Text":
                # Validate text data
                text = data.decode('utf-8')
                if not any(c.isprintable() for c in text):
                    raise ValueError("Invalid text data: no printable characters found")
                    
            elif data_type == "Audio":
                # Validate audio data with marker
                if len(data) < 32:  # Minimum size needed for our format
                    raise ValueError("Invalid audio data: too small")
                if not data.startswith(b'WAVMARK'):
                    raise ValueError("Invalid audio data: missing marker")
                    
            elif data_type == "Image":
                # Validate image data
                if len(data) < 8:  # Minimum size for dimensions
                    raise ValueError("Invalid image data: too small")
                width, height = struct.unpack('>II', data[:8])
                if width <= 0 or height <= 0 or width > 32000 or height > 32000:
                    raise ValueError(f"Invalid image dimensions: {width}x{height}")
                if len(data) < 8 + width * height * 3:
                    raise ValueError("Invalid image data: incomplete pixel data")
                    
        except ValueError as e:
            raise ValueError(f"Invalid {data_type.lower()} data: {str(e)}")
        except Exception as e:
            raise ValueError(f"Data validation failed: {str(e)}")

    async def save_image_async(self, image, output_path):
        """Asynchronous image saving with WebP format"""
        try:
            # Convert to WebP for better compression
            webp_path = str(Path(output_path).with_suffix('.webp'))
            
            # Convert to numpy array
            img_array = np.array(image)
            
            # Use OpenCV for faster processing
            img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
            
            # Save with WebP compression
            async with aiofiles.open(webp_path, 'wb') as f:
                is_success, buf = cv2.imencode(
                    ".webp", 
                    img_bgr,
                    [cv2.IMWRITE_WEBP_QUALITY, 75]
                )
                if not is_success:
                    raise ValueError("Failed to encode image")
                await f.write(buf.tobytes())
            
            return webp_path
            
        except Exception as e:
            raise ValueError(f"Failed to save image: {str(e)}")

    def process_image_async(self, image_path):
        """Asynchronous image processing with GPU acceleration"""
        try:
            # Use memory mapping for large files
            file_size = os.path.getsize(image_path)
            
            with open(image_path, 'rb') as f:
                mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
                
                # Process image with GPU if available
                if HAS_GPU and HAS_CUDA:
                    try:
                        # Load image using OpenCV for CUDA support
                        img_array = cv2.imdecode(np.frombuffer(mm, dtype=np.uint8), cv2.IMREAD_COLOR)
                        if img_array is None:
                            raise ValueError("Failed to load image")
                        
                        # Convert BGR to RGB
                        img_array = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)
                        
                        # Transfer to GPU
                        gpu_array = cp.asarray(img_array)
                        
                        # Process on GPU
                        if file_size > 50 * 1024 * 1024:  # 50MB
                            # For large images, process in chunks
                            chunk_size = 16 * 1024 * 1024  # 16MB chunks
                            height, width = img_array.shape[:2]
                            
                            for y in range(0, height, chunk_size // width // 3):
                                chunk_end = min(y + chunk_size // width // 3, height)
                                chunk = gpu_array[y:chunk_end]
                                # Process chunk
                                chunk = self.process_gpu_chunk(chunk)
                                gpu_array[y:chunk_end] = chunk
                        else:
                            # Process entire image at once
                            gpu_array = self.process_gpu_chunk(gpu_array)
                        
                        # Transfer back to CPU
                        processed_array = cp.asnumpy(gpu_array)
                        
                    except Exception as e:
                        print(f"GPU processing failed: {str(e)}, falling back to CPU")
                        # Fall back to CPU processing
                        processed_array = np.array(Image.open(image_path).convert('RGB'))
                else:
                    # CPU processing with optimized libraries
                    img = Image.open(image_path).convert('RGB')
                    processed_array = np.array(img)
                
                # Create PIL Image from array
                processed_image = Image.fromarray(processed_array)
                
                mm.close()
                return processed_image
                
        except Exception as e:
            raise ValueError(f"Image processing failed: {str(e)}")

    def optimize_thread_pool(self):
        """Optimize thread pool size based on system resources"""
        try:
            # Get physical CPU cores
            cpu_count = psutil.cpu_count(logical=False) or 4
            
            # Get available memory
            memory_gb = psutil.virtual_memory().total / (1024**3)
            
            # Calculate optimal thread count
            # Use more threads for systems with more memory
            memory_factor = min(4, max(1, int(memory_gb / 4)))
            optimal_threads = min(
                32,  # Maximum practical limit
                cpu_count * memory_factor,  # Scale with CPU and memory
                int(memory_gb * 2)  # Memory-based limit
            )
            
            # Set thread pool size
            self.max_workers = optimal_threads
            
            # Configure chunk sizes based on memory
            self.chunk_size = min(
                8 * 1024 * 1024,  # 8MB maximum
                max(16 * 1024, int(memory_gb * 1024 * 1024 / optimal_threads))  # Scale with memory
            )
            
            return optimal_threads
            
        except Exception as e:
            print(f"Error optimizing thread pool: {str(e)}")
            return 4  # Fallback to conservative value

    def run(self):
        self.window.mainloop()

if __name__ == "__main__":
    app = SteganographyApp()
    app.run()
    app.run()