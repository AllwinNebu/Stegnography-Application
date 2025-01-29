import os
import customtkinter as ctk
from tkinter import messagebox
from Crypto.Cipher import AES
from Crypto.Util.Padding import pad, unpad
from Crypto.Random import get_random_bytes
from PIL import Image
import numpy as np
from pathlib import Path
import zlib

class SteganographyApp:
    def __init__(self):
        self.setup_gui()
        
    def setup_gui(self):
        # Configure appearance
        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("blue")
        
        # Create main window
        self.window = ctk.CTk()
        self.window.title("Steganography Tool")
        self.window.geometry("800x600")
        
        # Create tabs
        self.tabview = ctk.CTkTabview(self.window)
        self.tabview.pack(padx=20, pady=20, fill="both", expand=True)
        
        # Add tabs
        self.tab_hide = self.tabview.add("Hide Data")
        self.tab_extract = self.tabview.add("Extract Data")
        
        self.setup_hide_tab()
        self.setup_extract_tab()
        
    def setup_hide_tab(self):
        # File selection frame
        file_frame = ctk.CTkFrame(self.tab_hide)
        file_frame.pack(padx=20, pady=20, fill="x")
        
        # Image selection
        ctk.CTkLabel(file_frame, text="Carrier Image:").pack(anchor="w")
        self.image_path = ctk.CTkEntry(file_frame, width=400)
        self.image_path.pack(side="left", padx=5)
        ctk.CTkButton(file_frame, text="Browse", command=self.browse_image).pack(side="left", padx=5)
        
        # Data type selection
        data_frame = ctk.CTkFrame(self.tab_hide)
        data_frame.pack(padx=20, pady=20, fill="x")
        
        self.data_type = ctk.CTkSegmentedButton(
            data_frame,
            values=["Text", "File", "Audio"],
            command=self.toggle_data_input
        )
        self.data_type.pack(pady=10)
        self.data_type.set("Text")
        
        # Text input
        self.text_frame = ctk.CTkFrame(data_frame)
        self.text_frame.pack(fill="x", pady=10)
        self.text_input = ctk.CTkTextbox(self.text_frame, height=100)
        self.text_input.pack(fill="x", padx=5)
        
        # File input
        self.file_frame = ctk.CTkFrame(data_frame)
        self.file_path = ctk.CTkEntry(self.file_frame, width=400)
        self.file_path.pack(side="left", padx=5)
        ctk.CTkButton(self.file_frame, text="Browse", command=self.browse_file).pack(side="left", padx=5)
        
        # Key input
        key_frame = ctk.CTkFrame(self.tab_hide)
        key_frame.pack(padx=20, pady=20, fill="x")
        ctk.CTkLabel(key_frame, text="Encryption Key:").pack(anchor="w")
        self.key_input = ctk.CTkEntry(key_frame, show="*")
        self.key_input.pack(fill="x", padx=5)
        
        # Hide button
        self.hide_button = ctk.CTkButton(self.tab_hide, text="Hide Data", command=self.hide_data)
        self.hide_button.pack(pady=20)
        
    def setup_extract_tab(self):
        # File selection
        file_frame = ctk.CTkFrame(self.tab_extract)
        file_frame.pack(padx=20, pady=20, fill="x")
        
        ctk.CTkLabel(file_frame, text="Stego Image:").pack(anchor="w")
        self.extract_image_path = ctk.CTkEntry(file_frame, width=400)
        self.extract_image_path.pack(side="left", padx=5)
        ctk.CTkButton(file_frame, text="Browse", command=self.browse_extract_image).pack(side="left", padx=5)
        
        # Key input
        key_frame = ctk.CTkFrame(self.tab_extract)
        key_frame.pack(padx=20, pady=20, fill="x")
        ctk.CTkLabel(key_frame, text="Encryption Key:").pack(anchor="w")
        self.extract_key_input = ctk.CTkEntry(key_frame, show="*")
        self.extract_key_input.pack(fill="x", padx=5)
        
        # Extract button
        self.extract_button = ctk.CTkButton(self.tab_extract, text="Extract Data", command=self.extract_data)
        self.extract_button.pack(pady=20)
        
        # Output
        self.output_text = ctk.CTkTextbox(self.tab_extract, height=200)
        self.output_text.pack(padx=20, pady=20, fill="both", expand=True)
        
    def toggle_data_input(self, value):
        if value == "Text":
            self.file_frame.pack_forget()
            self.text_frame.pack(fill="x", pady=10)
        else:
            self.text_frame.pack_forget()
            self.file_frame.pack(fill="x", pady=10)
    
    def browse_image(self):
        file_path = ctk.filedialog.askopenfilename(filetypes=[("Image files", "*.png *.jpg *.jpeg")])
        if file_path:
            self.image_path.delete(0, "end")
            self.image_path.insert(0, file_path)
    
    def browse_file(self):
        file_path = ctk.filedialog.askopenfilename()
        if file_path:
            self.file_path.delete(0, "end")
            self.file_path.insert(0, file_path)
    
    def browse_extract_image(self):
        file_path = ctk.filedialog.askopenfilename(filetypes=[("Image files", "*.png *.jpg *.jpeg")])
        if file_path:
            self.extract_image_path.delete(0, "end")
            self.extract_image_path.insert(0, file_path)
    
    def hide_data(self):
        try:
            image_path = self.image_path.get()
            key = self.key_input.get()
            
            if not image_path or not key:
                raise ValueError("Please provide both image path and encryption key")
            
            # Generate output path
            output_path = str(Path(image_path).parent / f"stego_{Path(image_path).name}")
            
            if self.data_type.get() == "Text":
                text = self.text_input.get("1.0", "end-1c")
                if not text:
                    raise ValueError("Please enter text to hide")
                self.hide_text_in_image(image_path, text, key, output_path)
            elif self.data_type.get() == "File":
                file_path = self.file_path.get()
                if not file_path:
                    raise ValueError("Please select a file to hide")
                self.hide_file_in_image(image_path, file_path, key, output_path)
            elif self.data_type.get() == "Audio":
                file_path = self.file_path.get()
                if not file_path:
                    raise ValueError("Please select an audio file to hide")
                self.hide_audio_in_image(image_path, file_path, key, output_path)
                
            messagebox.showinfo("Success", f"Data hidden successfully in {output_path}")
            
        except Exception as e:
            messagebox.showerror("Error", str(e))
    
    def extract_data(self):
        try:
            image_path = self.extract_image_path.get()
            key = self.extract_key_input.get()
            
            if not image_path or not key:
                raise ValueError("Please provide both image path and encryption key")
            
            # Try to extract as text first
            try:
                text = self.extract_text_from_image(image_path, key)
                self.output_text.delete("1.0", "end")
                self.output_text.insert("1.0", text)
            except:
                # If text extraction fails, try file extraction
                output_path = ctk.filedialog.asksaveasfilename()
                if output_path:
                    self.extract_file_from_image(image_path, key, output_path)
                    self.output_text.delete("1.0", "end")
                    self.output_text.insert("1.0", f"File extracted to: {output_path}")
                
        except Exception as e:
            messagebox.showerror("Error", str(e))
    
    # Modified AES (MAES) Functions
    def generate_key(self, shared_key):
        return shared_key.zfill(32).encode('utf-8')

    def encrypt_maes(self, data, key):
        cipher = AES.new(key, AES.MODE_ECB)
        return cipher.encrypt(pad(data, AES.block_size))

    def decrypt_maes(self, data, key):
        cipher = AES.new(key, AES.MODE_ECB)
        return unpad(cipher.decrypt(data), AES.block_size)

    def compress_data(self, data):
        return zlib.compress(data)

    def decompress_data(self, data):
        return zlib.decompress(data)

    def embed_data_in_image(self, image_path, data, output_path):
        # Compress the data first for large data
        compressed_data = self.compress_data(data)

        # Convert data to binary string
        binary_data = ''.join(format(byte, '08b') for byte in compressed_data)
        data_len = len(binary_data)
        
        # Prepare image
        image = Image.open(image_path)
        image_array = np.array(image)
        
        # Check if image can hold the data
        max_bits = image_array.size * 3  # Three channels per pixel (RGB)
        if len(binary_data) > max_bits:
            raise ValueError(f"Data is too large. Image can only hide {max_bits // 8} bytes")

        # Embed data in higher bits (MSB) of the pixel channels
        idx = 0
        flat_array = image_array.flatten()
        
        # Embed length
        length_binary = format(len(binary_data), '032b')
        for i in range(32):
            flat_array[idx] = (flat_array[idx] & 0xFE) | (int(length_binary[i]) << 7)  # Embed in MSB
            idx += 1
        
        # Embed data in the remaining bits
        for bit in binary_data:
            flat_array[idx] = (flat_array[idx] & 0xFE) | (int(bit) << 7)  # Embed in MSB
            idx += 1
        
        # Save modified image
        modified_array = flat_array.reshape(image_array.shape)
        modified_image = Image.fromarray(modified_array)
        modified_image.save(output_path, format='PNG')
    
    def extract_data_from_image(self, image_path):
        # Load image
        image = Image.open(image_path)
        flat_array = np.array(image).flatten()
        
        # Extract length
        length_binary = ''.join(str(pixel & 1) for pixel in flat_array[:32])
        data_length = int(length_binary, 2)
        
        # Extract data
        binary_data = ''.join(str(pixel & 1) for pixel in flat_array[32:32 + data_length * 8])
        
        # Convert to bytes
        data = bytearray()
        for i in range(0, len(binary_data), 8):
            byte = int(binary_data[i:i+8], 2)
            data.append(byte)
        
        return bytes(data)
    
    def hide_text_in_image(self, image_path, text, key, output_path):
        data = text.encode('utf-8')
        key = self.generate_key(key)
        encrypted_data = self.encrypt_maes(data, key)
        self.embed_data_in_image(image_path, encrypted_data, output_path)
    
    def hide_file_in_image(self, image_path, file_path, key, output_path):
        with open(file_path, 'rb') as file:
            data = file.read()
        key = self.generate_key(key)
        encrypted_data = self.encrypt_maes(data, key)
        self.embed_data_in_image(image_path, encrypted_data, output_path)

    def hide_audio_in_image(self, image_path, audio_path, key, output_path):
        with open(audio_path, 'rb') as audio_file:
            data = audio_file.read()
        key = self.generate_key(key)
        encrypted_data = self.encrypt_maes(data, key)
        self.embed_data_in_image(image_path, encrypted_data, output_path)
    
    def extract_text_from_image(self, image_path, key):
        encrypted_data = self.extract_data_from_image(image_path)
        key = self.generate_key(key)
        decrypted_data = self.decrypt_maes(encrypted_data, key)
        decompressed_data = self.decompress_data(decrypted_data)
        return decompressed_data.decode('utf-8')
    
    def extract_file_from_image(self, image_path, key, output_path):
        encrypted_data = self.extract_data_from_image(image_path)
        key = self.generate_key(key)
        decrypted_data = self.decrypt_maes(encrypted_data, key)
        decompressed_data = self.decompress_data(decrypted_data)
        with open(output_path, 'wb') as file:
            file.write(decompressed_data)

    def run(self):
        self.window.mainloop()

if __name__ == "__main__":
    app = SteganographyApp()
    app.run()
