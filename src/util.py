import tkinter as tk

class Rectangle:
    def __init__(self, canvas, x1, y1, x2, y2, color='black'):
        self.canvas = canvas
        self.x1 = x1
        self.y1 = y1
        self.x2 = x2
        self.y2 = y2
        self.color = color
        self.rect = None

    def draw(self):
        self.rect = self.canvas.create_rectangle(self.x1, self.y1, self.x2, self.y2, fill=self.color)

    def move(self, dx, dy):
        self.canvas.move(self.rect, dx, dy)

    def delete(self):
        self.canvas.delete(self.rect)

class CanvasApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Rectangle Drawing App")

        self.canvas = tk.Canvas(root, width=400, height=400)
        self.canvas.pack()

        self.rectangles = []

        self.canvas.bind("<Button-1>", self.on_canvas_click)
        self.canvas.bind("<Button-3>", self.on_canvas_right_click)

    def on_canvas_click(self, event):
        x1, y1 = event.x, event.y
        x2, y2 = x1 + 50, y1 + 50
        rectangle = Rectangle(self.canvas, x1, y1, x2, y2, 'blue')
        rectangle.draw()
        self.rectangles.append(rectangle)

    def on_canvas_right_click(self, event):
        if self.rectangles:
            last_rectangle = self.rectangles.pop()
            last_rectangle.delete()

if __name__ == "__main__":
    root = tk.Tk()
    app = CanvasApp(root)
    root.mainloop()