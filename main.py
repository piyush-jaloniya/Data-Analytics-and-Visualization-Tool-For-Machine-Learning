from app import MLApp
import ttkbootstrap as tb
import tkinter as tk


if __name__ == "__main__":
    # root= tk.Tk()
    root = tb.Window(themename="pulse")
    # root = tb.Window(themename="solar")
    # root = tb.Window(themename="darkly")
    # root = tb.Window(themename="yeti")
    # root = tb.Window(themename="sandstone")
    # root = tb.Window(themename="flatly")  
    app = MLApp(root)
    root.mainloop()