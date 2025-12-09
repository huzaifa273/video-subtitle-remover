import cv2
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from PIL import Image, ImageTk
import logging
import os
import sys
import configparser
import threading
import multiprocessing

# --- IMPORT BACKEND ---
# We need to make sure Python can find the 'backend' folder
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
# Try importing the backend module from the repo
try:
    import backend.main
except ImportError:
    print("Error: Could not find 'backend' folder. Please ensure this script is inside the 'video-subtitle-remover' directory.")
    # We define a dummy placeholder so the code doesn't crash immediately if backend is missing
    backend = None 

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class SubtitleRemoverApp:
    def __init__(self, root):
        self.root = root
        self.root.title(f"Video Subtitle Remover (Tkinter) - v{getattr(backend.main.config, 'VERSION', '1.0') if backend else '1.0'}")
        
        # --- State Variables ---
        self.video_path = ""
        self.cap = None
        self.playing = False
        self.last_frame = None
        self.sr = None  # This will hold the backend SubtitleRemover instance
        
        self.frame_width = 0
        self.frame_height = 0
        self.fps = 0
        self.frame_count = 0
        
        self.subtitle_config_file = os.path.join(os.path.dirname(__file__), 'subtitle.ini')
        
        # Default Rectangle (will be overwritten by config)
        self.rect_x = 0
        self.rect_y = 0
        self.rect_w = 0
        self.rect_h = 0

        # UI Dimensions
        self.screen_width = self.root.winfo_screenwidth()
        self.screen_height = self.root.winfo_screenheight()
        # Cap window size to 960x540 (16:9) or smaller if screen is small
        self.window_width = min(960, self.screen_width - 100)
        self.window_height = int(self.window_width * 9 / 16)

        self.create_widgets()
        self.load_config() # Load saved rectangle position

    def create_widgets(self):
        # 1. Main Canvas Area
        self.canvas_frame = tk.Frame(self.root)
        self.canvas_frame.pack(fill="both", expand=True)

        self.canvas = tk.Canvas(self.canvas_frame, width=self.window_width, height=self.window_height, bg="black")
        self.canvas.pack(pady=10)

        # 2. Controls Area
        self.controls_frame = tk.Frame(self.root)
        self.controls_frame.pack(fill="x", padx=10, pady=5)

        # File Controls
        self.btn_open = tk.Button(self.controls_frame, text="Open Video", command=self.open_video)
        self.btn_open.grid(row=0, column=0, padx=5)

        self.btn_run = tk.Button(self.controls_frame, text="RUN REMOVAL", command=self.run_removal, bg="#ffcccc", state="disabled")
        self.btn_run.grid(row=0, column=1, padx=5)
        
        # Progress Bar
        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(self.controls_frame, variable=self.progress_var, maximum=100)
        self.progress_bar.grid(row=0, column=2, sticky="ew", padx=10)
        self.controls_frame.columnconfigure(2, weight=1) # Make progress bar expand

        # 3. Sliders Area (Frame for positioning)
        self.sliders_frame = tk.LabelFrame(self.root, text="Subtitle Area Selection")
        self.sliders_frame.pack(fill="x", padx=10, pady=5)

        # X Position
        tk.Label(self.sliders_frame, text="X Pos").grid(row=0, column=0)
        self.x_slider = tk.Scale(self.sliders_frame, from_=0, to=self.window_width, orient="horizontal", command=self.update_preview)
        self.x_slider.grid(row=0, column=1, sticky="ew")

        # Width
        tk.Label(self.sliders_frame, text="Width").grid(row=0, column=2)
        self.w_slider = tk.Scale(self.sliders_frame, from_=0, to=self.window_width, orient="horizontal", command=self.update_preview)
        self.w_slider.grid(row=0, column=3, sticky="ew")

        # Y Position
        tk.Label(self.sliders_frame, text="Y Pos").grid(row=1, column=0)
        self.y_slider = tk.Scale(self.sliders_frame, from_=0, to=self.window_height, orient="horizontal", command=self.update_preview)
        self.y_slider.grid(row=1, column=1, sticky="ew")

        # Height
        tk.Label(self.sliders_frame, text="Height").grid(row=1, column=2)
        self.h_slider = tk.Scale(self.sliders_frame, from_=0, to=self.window_height, orient="horizontal", command=self.update_preview)
        self.h_slider.grid(row=1, column=3, sticky="ew")
        
        # Video Navigation Slider
        self.nav_slider = tk.Scale(self.root, from_=0, to=100, orient="horizontal", label="Video Navigation", command=self.seek_video)
        self.nav_slider.pack(fill="x", padx=10)

        self.sliders_frame.columnconfigure(1, weight=1)
        self.sliders_frame.columnconfigure(3, weight=1)

    def load_config(self):
        """ Read subtitle.ini to restore previous rectangle """
        if os.path.exists(self.subtitle_config_file):
            config = configparser.ConfigParser()
            config.read(self.subtitle_config_file)
            try:
                # These are percentages (0.0 to 1.0)
                yp = float(config['AREA']['Y'])
                hp = float(config['AREA']['H'])
                xp = float(config['AREA']['X'])
                wp = float(config['AREA']['W'])
                
                # We can't set slider values yet because we don't know window size relative to video
                # We store them to apply when video opens
                self.saved_ratios = (yp, hp, xp, wp)
            except:
                self.saved_ratios = (0.8, 0.15, 0.05, 0.9) # Defaults
        else:
            self.saved_ratios = (0.8, 0.15, 0.05, 0.9)

    def save_config(self, y_ratio, h_ratio, x_ratio, w_ratio):
        """ Save current rectangle ratios to subtitle.ini """
        with open(self.subtitle_config_file, 'w') as f:
            f.write('[AREA]\n')
            f.write(f'Y = {y_ratio}\n')
            f.write(f'H = {h_ratio}\n')
            f.write(f'X = {x_ratio}\n')
            f.write(f'W = {w_ratio}\n')

    def open_video(self):
        path = filedialog.askopenfilename(filetypes=[("Video Files", "*.mp4 *.mkv *.avi *.flv")])
        if not path: return
        
        self.video_path = path
        self.cap = cv2.VideoCapture(self.video_path)
        
        self.frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)

        # Enable sliders and Run button
        self.btn_run.config(state="normal")
        self.nav_slider.config(to=self.frame_count)

        # Apply saved config ratios to current UI sliders
        yp, hp, xp, wp = self.saved_ratios
        
        # Convert Ratios -> Preview Window Coordinates
        self.y_slider.set(yp * self.window_height)
        self.h_slider.set(hp * self.window_height)
        self.x_slider.set(xp * self.window_width)
        self.w_slider.set(wp * self.window_width)

        # Show first frame
        self.read_frame()
        self.draw_preview()

    def seek_video(self, value):
        if self.cap:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, int(value))
            self.read_frame()
            self.draw_preview()

    def read_frame(self):
        if self.cap:
            ret, frame = self.cap.read()
            if ret:
                self.last_frame = frame

    def update_preview(self, event=None):
        """ Called when rect sliders move """
        self.draw_preview()

    def draw_preview(self):
        if self.last_frame is None: return

        # 1. Resize original frame to Window Size
        frame_resized = cv2.resize(self.last_frame, (self.window_width, self.window_height))

        # 2. Get Slider Values (Window Coordinates)
        x = int(self.x_slider.get())
        y = int(self.y_slider.get())
        w = int(self.w_slider.get())
        h = int(self.h_slider.get())

        # 3. Draw Rectangle
        cv2.rectangle(frame_resized, (x, y), (x+w, y+h), (0, 255, 0), 2)

        # 4. Display
        img = Image.fromarray(cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB))
        imgtk = ImageTk.PhotoImage(image=img)
        self.canvas.create_image(0, 0, anchor="nw", image=imgtk)
        self.canvas.image = imgtk

    def run_removal(self):
        """ The bridge to the Backend Logic """
        if not self.video_path or backend is None:
            messagebox.showerror("Error", "Backend not found or video not loaded.")
            return

        # 1. Disable UI to prevent crashes
        self.btn_run.config(state="disabled", text="Processing...")
        self.btn_open.config(state="disabled")

        # 2. Calculate coordinates for the ORIGINAL video
        # We must scale up from Preview Size -> Original Video Size
        
        preview_x = self.x_slider.get()
        preview_y = self.y_slider.get()
        preview_w = self.w_slider.get()
        preview_h = self.h_slider.get()

        scale_x = self.frame_width / self.window_width
        scale_y = self.frame_height / self.window_height

        real_ymin = int(preview_y * scale_y)
        real_ymax = int((preview_y + preview_h) * scale_y)
        real_xmin = int(preview_x * scale_x)
        real_xmax = int((preview_x + preview_w) * scale_x)

        # Safety clamps
        real_ymax = min(real_ymax, self.frame_height)
        real_xmax = min(real_xmax, self.frame_width)

        subtitle_area = (real_ymin, real_ymax, real_xmin, real_xmax)
        print(f"Removal Area (Original Video Coords): {subtitle_area}")

        # 3. Save Config for next time
        self.save_config(
            real_ymin / self.frame_height,
            (real_ymax - real_ymin) / self.frame_height,
            real_xmin / self.frame_width,
            (real_xmax - real_xmin) / self.frame_width
        )

        # 4. Initialize Backend in a separate thread
        # We pass 'True' for use_gpu if available (change to False if issues arise)
        self.sr = backend.main.SubtitleRemover(self.video_path, subtitle_area, True)
        
        # Start the backend work in a thread
        t = threading.Thread(target=self.sr.run)
        t.daemon = True
        t.start()

        # 5. Start monitoring the progress
        self.monitor_progress()

    def monitor_progress(self):
        """ Check the backend status every 500ms """
        if self.sr:
            # Update Progress Bar
            # Assuming sr.progress_total is a percentage (0-100)
            current_prog = getattr(self.sr, 'progress_total', 0)
            self.progress_var.set(current_prog)
            
            # Check if backend is finished
            if getattr(self.sr, 'isFinished', False):
                messagebox.showinfo("Success", "Subtitle removal complete!")
                self.btn_run.config(state="normal", text="RUN REMOVAL")
                self.btn_open.config(state="normal")
                self.sr = None
                return

        # If not finished, check again in 500ms
        self.root.after(500, self.monitor_progress)

if __name__ == "__main__":
    # Required for multiprocessing support (used by backend)
    multiprocessing.freeze_support()
    
    root = tk.Tk()
    app = SubtitleRemoverApp(root)
    root.mainloop()