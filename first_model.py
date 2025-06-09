import numpy as np
import matplotlib.pyplot as plt
from scipy import fft
from scipy.io import wavfile
import cv2
from typing import Tuple, Optional
import numpy.typing as npt
import os

OUTPUT_DIR = "first_model_dir"
os.makedirs(OUTPUT_DIR, exist_ok=True)

class SignalProcessor:
    def __init__(self):
        self.original_signal = None
        self.noisy_signal = None
        self.filtered_signal = None
        self.sampling_rate = None
        self.time_points = None
    
    def load_wav_signal(self, file_path: str) -> None:
        try:
            self.sampling_rate, self.original_signal = wavfile.read(file_path)
            # Normalize signal to [-1, 1] (float32)
            self.original_signal = self.original_signal.astype(np.float32) / np.iinfo(np.int16).max
            self.time_points = np.arange(0, len(self.original_signal)) / self.sampling_rate
        except Exception as e:
            print(f"Error while loading file: {e}")
    
    def save_wav_signal(self, file_path: str, signal: Optional[np.ndarray] = None) -> None:
        if signal is None:
            signal = self.original_signal
        
        if signal is not None:
            signal_int = (signal * np.iinfo(np.int16).max).astype(np.int16)
            wavfile.write(file_path, self.sampling_rate, signal_int)
        else:
            raise ValueError("No signal to save")
    
    def add_noise(self, method: str = 'gaussian', noise_params: dict = None) -> None:
        if self.original_signal is None:
            raise ValueError("No signal loaded")
            
        if method == 'gaussian':
            noise = np.random.normal(noise_params.get('mean', 0),
                                   noise_params.get('std', 0.05),
                                   size=self.original_signal.shape)
            self.noisy_signal = self.original_signal + noise
            self.noisy_signal = np.clip(self.noisy_signal, -1.0, 1.0)
        else:
            raise ValueError(f"Unsupported noise method: {method}")

    def get_power_spectrum(self, signal: np.ndarray, frame_length: int) -> Tuple[np.ndarray, np.ndarray]:
        window = np.hanning(frame_length)
        windowed = signal * window
        spectrum = fft.rfft(windowed)
        power = np.abs(spectrum) ** 2
        freq = fft.rfftfreq(frame_length, 1/self.sampling_rate)
        return power, freq

    def filter_signal_by_window_analysis(self, filter_params: dict = None) -> None:
        if self.original_signal is None:
            raise ValueError("No signal loaded")

        frame_length = filter_params.get('frame_length', 4096)
        overlap = filter_params.get('overlap', 0.75)
        noise_reduction = filter_params.get('noise_reduction_koeff', 0.3)
        
        hop_length = int(frame_length * (1 - overlap))
        
        self.filtered_signal = np.zeros_like(self.original_signal)
        window_sum = np.zeros_like(self.original_signal)
        
        window = np.hanning(frame_length)
        
        n_noise_frames = 5
        noise_power = np.zeros(frame_length // 2 + 1)
        for i in range(n_noise_frames):
            if i * hop_length + frame_length <= len(self.original_signal):
                frame = self.original_signal[i * hop_length:i * hop_length + frame_length]
                power, _ = self.get_power_spectrum(frame, frame_length)
                noise_power += power / n_noise_frames
        
        for i in range(0, len(self.original_signal) - frame_length + 1, hop_length):
            frame = self.original_signal[i:i + frame_length]
            
            # spectrum of current frame
            power, freq = self.get_power_spectrum(frame, frame_length)
            
            # noise reduction mask
            snr = power / (noise_power + 1e-10)
            gain = np.maximum(1 - noise_reduction / np.maximum(snr, noise_reduction), 0.1)
            
            spectrum = fft.rfft(frame * window)
            filtered_spectrum = spectrum * gain
            
            # inverse transform
            filtered_frame = fft.irfft(filtered_spectrum)
            
            # Overlap-add
            self.filtered_signal[i:i + frame_length] += filtered_frame * window
            window_sum[i:i + frame_length] += window ** 2
        
        # normalization
        window_sum = np.maximum(window_sum, 1e-10)
        self.filtered_signal = self.filtered_signal / window_sum
        
        self.filtered_signal = np.clip(self.filtered_signal, -1.0, 1.0)
        
        # normalization of amplitude with preserving dynamic range
        target_rms = np.sqrt(np.mean(self.original_signal ** 2))
        current_rms = np.sqrt(np.mean(self.filtered_signal ** 2))
        if current_rms > 0:
            self.filtered_signal *= target_rms / current_rms

    def kuwahara_filter(self, image: np.ndarray, window_size: int = 5) -> np.ndarray:
        if window_size % 2 == 0:
            window_size = window_size + 1
        
        radius = window_size // 2
        rows, cols, channels = image.shape
        output = np.zeros_like(image, dtype=np.float32)
        
        padded = cv2.copyMakeBorder(image.astype(np.float32), radius, radius, radius, radius, cv2.BORDER_REFLECT)
        
        total_pixels = rows * cols
        processed_pixels = 0
        print("\nApplying Kuwahara filter:")
        
        for i in range(radius, rows + radius):
            for j in range(radius, cols + radius):
                # define 4 quadrants
                q1 = padded[i-radius:i+1, j-radius:j+1]
                q2 = padded[i-radius:i+1, j:j+radius+1]
                q3 = padded[i:i+radius+1, j-radius:j+1]
                q4 = padded[i:i+radius+1, j:j+radius+1]
                
                # calculate mean and variance for each quadrant
                means = []
                variances = []
                
                for q in [q1, q2, q3, q4]:
                    means.append(np.mean(q, axis=(0,1)))
                    variances.append(np.mean(np.var(q, axis=(0,1))))
                
                # select quadrant with minimum variance
                min_var_idx = np.argmin(variances)
                output[i-radius, j-radius] = means[min_var_idx]
                
                processed_pixels += 1
                if processed_pixels % (total_pixels // 10) == 0:
                    progress = (processed_pixels / total_pixels) * 100
                    print(f"Progress: {progress:.1f}%")
        
        print("Kuwahara filter completed")
        return output.astype(np.uint8)

    def anisotropic_diffusion(self, image: np.ndarray, num_iter: int = 10, kappa: float = 30, gamma: float = 0.15) -> np.ndarray:
        image = image.astype(np.float32)
        
        # conductivity function
        def g(x, k):
            return np.exp(-(x/k)**2)
        
        # blur channel
        filtered = np.zeros_like(image)
        print("\n Applying anisotropic diffusion:")
        
        for channel in range(image.shape[2]):
            print(f"\nProcessing channel {channel + 1}/{image.shape[2]}")
            u = image[:,:,channel].copy()
            
            # iterative diffusion process
            for iter_num in range(num_iter):
                # gradients in 4 directions
                nabla_n = np.roll(u, -1, axis=0) - u
                nabla_s = np.roll(u, 1, axis=0) - u
                nabla_e = np.roll(u, 1, axis=1) - u
                nabla_w = np.roll(u, -1, axis=1) - u
                
                # conductivity coefficients
                cn = g(nabla_n, kappa)
                cs = g(nabla_s, kappa)
                ce = g(nabla_e, kappa)
                cw = g(nabla_w, kappa)
                
                u += gamma * (cn*nabla_n + cs*nabla_s + ce*nabla_e + cw*nabla_w)
                
                progress = ((iter_num + 1) / num_iter) * 100
                print(f"Iteration {iter_num + 1}/{num_iter} - {progress:.1f}%")
            
            filtered[:,:,channel] = u
        
        print("Anisotropic diffusion completed")
        return np.clip(filtered, 0, 255).astype(np.uint8)

    def process_image(self, image_path: str, noise_std: float = 15) -> Tuple[npt.NDArray, npt.NDArray, npt.NDArray]:
        print("\nStarting image processing...")
        
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError("Failed to load image")
        
        print("Adding noise...")
        noisy = np.clip(image + np.random.normal(0, noise_std, image.shape), 0, 255).astype(np.uint8)
        
        kuwahara = self.kuwahara_filter(noisy, window_size=5)
        filtered = self.anisotropic_diffusion(kuwahara, num_iter=3, kappa=50, gamma=0.15)
        
        print("Image processing completed!")
        return image, noisy, filtered
    
    def plot_images(self, original: npt.NDArray, noisy: npt.NDArray, filtered: npt.NDArray) -> None:
        plt.figure(figsize=(15, 5))
        
        plt.subplot(131)
        plt.imshow(cv2.cvtColor(original, cv2.COLOR_BGR2RGB))
        plt.title('Original image')
        plt.axis('off')
        
        plt.subplot(132)
        plt.imshow(cv2.cvtColor(noisy, cv2.COLOR_BGR2RGB))
        plt.title('Noisy image')
        plt.axis('off')
        
        plt.subplot(133)
        plt.imshow(cv2.cvtColor(filtered, cv2.COLOR_BGR2RGB))
        plt.title('Filtered image')
        plt.axis('off')
        
        plt.tight_layout()
        plt.show()

    def plot_signals(self) -> None:
        if self.time_points is None or self.original_signal is None:
            raise ValueError("No signals loaded to plot")
            
        plt.figure(figsize=(15, 10))
        
        plt.subplot(3, 1, 1)
        plt.plot(self.time_points, self.original_signal)
        plt.title('Original Signal')
        plt.xlabel('Time (s)')
        plt.ylabel('Amplitude')
        plt.grid(True)
        
        if self.noisy_signal is not None:
            plt.subplot(3, 1, 2)
            plt.plot(self.time_points, self.noisy_signal)
            plt.title('Noisy Signal')
            plt.xlabel('Time (s)')
            plt.ylabel('Amplitude')
            plt.grid(True)
        
        if self.filtered_signal is not None:
            plt.subplot(3, 1, 3)
            plt.plot(self.time_points, self.filtered_signal)
            plt.title('Filtered Signal')
            plt.xlabel('Time (s)')
            plt.ylabel('Amplitude')
            plt.grid(True)
        
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, 'signal_plots.png'))
        plt.show()

def create_test_signal(duration: float = 1.0, sampling_rate: int = 44100) -> Tuple[np.ndarray, np.ndarray]:
    t = np.linspace(0, duration, int(sampling_rate * duration))
    signal = (np.sin(2 * np.pi * 440 * t) +
             0.5 * np.sin(2 * np.pi * 880 * t) +
             0.25 * np.sin(2 * np.pi * 1760 * t))
    return t, signal

def main():
    processor = SignalProcessor()

    # print("Processing audio signal...")
    
    # t, test_signal = create_test_signal(duration=2.0)
    
    # test_signal_path = os.path.join(OUTPUT_DIR, "original_signal.wav")
    # processor.sampling_rate = 44100
    # processor.original_signal = test_signal
    # processor.time_points = t
    # processor.save_wav_signal(test_signal_path)
    
    # processor.load_wav_signal(test_signal_path)
    # processor.add_noise(method='gaussian', noise_params={'mean': 0, 'std': 0.05})
    
    # noisy_signal_path = os.path.join(OUTPUT_DIR, "noisy_signal.wav")
    # processor.save_wav_signal(noisy_signal_path, processor.noisy_signal)
    
    # processor.filter_signal_by_window_analysis(filter_params={
    #     'frame_length': 4096,
    #     'overlap': 0.75,
    #     'noise_reduction_koeff': 0.3
    # })
    
    # filtered_signal_path = os.path.join(OUTPUT_DIR, "filtered_signal.wav")
    # processor.save_wav_signal(filtered_signal_path, processor.filtered_signal)
    
    print("\nProcessing signal_11.npy...")
    try:
        signal_data = np.load('first_model_dir/signal_11.npy')
        
        signal_max = np.max(np.abs(signal_data))
        signal_data = signal_data / signal_max
        
        processor.original_signal = signal_data.copy()
        processor.sampling_rate = 44100
        processor.time_points = np.arange(len(signal_data)) / processor.sampling_rate
        
        signal11_wav_path = os.path.join(OUTPUT_DIR, "signal11_original.wav")
        processor.save_wav_signal(signal11_wav_path)
        
        processor.add_noise(method='gaussian', noise_params={'mean': 0, 'std': 0.1})
       
        noisy_signal11_wav_path = os.path.join(OUTPUT_DIR, "signal11_noisy.wav")
        processor.save_wav_signal(noisy_signal11_wav_path, processor.noisy_signal)
        
        processor.original_signal = processor.noisy_signal.copy()
        processor.filter_signal_by_window_analysis(filter_params={
            'frame_length': 4096,
            'overlap': 0.75,
            'noise_reduction_koeff': 0.3
        })
        
        filtered_signal11_wav_path = os.path.join(OUTPUT_DIR, "signal11_filtered.wav")
        processor.save_wav_signal(filtered_signal11_wav_path, processor.filtered_signal)
        
        original_signal = signal_data.copy()
        noisy_signal = processor.noisy_signal.copy()
        filtered_signal = processor.filtered_signal.copy()
        
        plt.figure(figsize=(15, 10))
        plt.suptitle('Signal Processing Results', fontsize=16)
        
        plt.subplot(3, 1, 1)
        plt.plot(processor.time_points, original_signal)
        plt.title(f'Original Signal (normalized, max abs: {signal_max:.2f})')
        plt.xlabel('Time (s)')
        plt.ylabel('Amplitude')
        plt.grid(True)
        
        plt.subplot(3, 1, 2)
        plt.plot(processor.time_points, noisy_signal)
        plt.title('Noisy Signal')
        plt.xlabel('Time (s)')
        plt.ylabel('Amplitude')
        plt.grid(True)
        
        plt.subplot(3, 1, 3)
        plt.plot(processor.time_points, filtered_signal)
        plt.title('Filtered Signal')
        plt.xlabel('Time (s)')
        plt.ylabel('Amplitude')
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, 'signal_plots.png'))
        plt.show()
        
        plt.figure(figsize=(15, 6))
        plt.plot(processor.time_points, noisy_signal, 'r', alpha=0.6, label='Noisy Signal')
        plt.plot(processor.time_points, original_signal, 'g', alpha=0.6, label='Original Signal')
        plt.plot(processor.time_points, filtered_signal, 'b', alpha=0.8, label='Filtered Signal')
        
        plt.title('Signal Comparison (Overlapped)')
        plt.xlabel('Time (s)')
        plt.ylabel('Amplitude')
        plt.grid(True)
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, 'signals_overlapped.png'))
        plt.show()
        
        print("\nSignal11 processing completed!")
        print(f"Files saved:")
        print("\nWAV files:")
        print(f"- {signal11_wav_path}")
        print(f"- {noisy_signal11_wav_path}")
        print(f"- {filtered_signal11_wav_path}")
        print("\nPlots:")
        print("- signal_plots.png")
        print("- signals_overlapped.png")
        
    except FileNotFoundError:
        print("signal11.npy file not found in first_model_dir!")
    except Exception as e:
        print(f"Error processing signal11.npy: {e}")

    print("\nProcessing image...")
    
    image_path = os.path.join(OUTPUT_DIR, "example.jpg")
    
    if os.path.exists(image_path):
        original, noisy, filtered = processor.process_image(
            image_path,
            noise_std=25
        )
        
        cv2.imwrite(os.path.join(OUTPUT_DIR, "noisy_image.png"), noisy)
        cv2.imwrite(os.path.join(OUTPUT_DIR, "filtered_image.png"), filtered)
        
        processor.plot_images(original, noisy, filtered)
    else:
        print(f"File {image_path} not found. Skipping image processing.")
    
    print("\nDone! Files saved in directory", OUTPUT_DIR)
    print("Audio:")
    # print(f"- {test_signal_path}")
    # print(f"- {noisy_signal_path}")
    # print(f"- {filtered_signal_path}")

    print(f"- {signal11_wav_path}")
    print(f"- {noisy_signal11_wav_path}")
    print(f"- {filtered_signal11_wav_path}")
    
    if os.path.exists(image_path):
        print("Images:")
        print(f"- {image_path}")
        print("- noisy_image.png")
        print("- filtered_image.png")

if __name__ == "__main__":
    main()
