from tkinter import *
from tkinter import filedialog

from PIL import Image, ImageTk
import io
from inpainting import harmonic_inpainting


class ImageInpainter(Frame):
    """Handwritten digits classifier class"""

    def __init__(self, parent):
        Frame.__init__(self, parent)
        self.parent = parent
        self.color = "green"
        self.brush_size = 12
        self.setUI()

    def set_color(self, new_color):
        """Additional brush color change"""
        self.color = new_color

    def set_brush_size(self, new_size):
        """Changes brush size for testing different lines width"""
        self.brush_size = new_size

    def draw(self, event):
        """Method to draw"""
        self.canv.create_oval(event.x - self.brush_size,
                              event.y - self.brush_size,
                              event.x + self.brush_size,
                              event.y + self.brush_size,
                              fill=self.color, outline=self.color)



    def save(self):
        """Save the current canvas state as the postscript
        uses classify method and shows the result"""
        self.canv.delete(self.backfround_image)
        self.canv.update()
        # print("And here: ", self.canv.image.height(), self.canv.image.width())
        ps = self.canv.postscript(colormode='mono', width=self.canv.image.width(), height=self.canv.image.height())
        img = Image.open(io.BytesIO(ps.encode('utf-8')))
        img = img.resize((self.canv.image.width(), self.canv.image.height()), Image.ANTIALIAS)
        # print(img.size)
        img.save('mask.png')
        self.canv.delete("all")
        self.canv.update()
        fidelity = 10
        tol = 1e-5
        maxiter = 500
        dt = 0.1
        harmonic_inpainting(self.file_path, 'mask.png', 'result.png', fidelity, tol, maxiter, dt)
        self.canv.image = ImageTk.PhotoImage(Image.open('result.png'))
        self.canv.create_image(0, 0, image=self.canv.image, anchor="nw")


    def setImage(self):
        self.file_path = filedialog.askopenfilename()
        self.canv.image = ImageTk.PhotoImage(Image.open(self.file_path))
        # print(self.canv.image.height(), self.canv.image.width())
        self.backfround_image = self.canv.create_image(0, 0, image=self.canv.image, anchor="nw")
        # self.parent.geometry(f"{self.canv.image.width()}x{self.canv.image.height()}")
        # self.parent.resizable(0, 0)

    def setUI(self):
        """Setup for all UI elements"""
        self.parent.title("Image Inpainting")
        self.pack(fill=BOTH, expand=1)
        self.columnconfigure(6, weight=1)
        self.rowconfigure(2, weight=1)
        self.canv = Canvas(self, bg='white')
        self.canv.grid(row=2, column=0, columnspan=7,
                       padx=5, pady=5,
                       sticky=E + W + S + N)
        self.canv.bind("<B1-Motion>",
                       self.draw)
        color_lab = Label(self, text="Color: ")
        color_lab.grid(row=0, column=0,
                       padx=6)
        black_btn = Button(self, text="Black", width=10,
                           command=lambda: self.set_color("black"))
        black_btn.grid(row=0, column=2)
        white_btn = Button(self, text="White", width=10,
                           command=lambda: self.set_color("white"))
        white_btn.grid(row=0, column=3)
        clear_btn = Button(self, text="Clear all", width=10,
                           command=lambda: self.canv.delete("all"))  # "all"
        clear_btn.grid(row=0, column=4, sticky=W)
        file_btn = Button(self, text="File", width=10,
                          command=lambda: self.setImage())
        file_btn.grid(row=0, column=5, sticky=W)
        size_lab = Label(self, text="Brush size: ")
        size_lab.grid(row=1, column=0, padx=5)
        five_btn = Button(self, text="one", width=10,
                          command=lambda: self.set_brush_size(1))
        five_btn.grid(row=1, column=2)
        seven_btn = Button(self, text="three", width=10,
                           command=lambda: self.set_brush_size(3))
        seven_btn.grid(row=1, column=3)
        ten_btn = Button(self, text="Twenty", width=10,
                         command=lambda: self.set_brush_size(20))
        ten_btn.grid(row=1, column=4)
        done_btn = Button(self, text="Done", width=10,
                          command=lambda: self.save())
        done_btn.grid(row=1, column=5)


def main():
    root = Tk()
    width = root.winfo_screenwidth()
    height = root.winfo_screenheight()
    root.geometry(f'{width}x{height}')
    # root.geometry("400x400")
    # root.resizable(0, 0)
    app = ImageInpainter(root)
    app.mainloop()


if __name__ == '__main__':
    main()
