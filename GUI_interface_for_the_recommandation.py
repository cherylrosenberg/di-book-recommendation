import tkinter as tk
from tkinter import messagebox
from movie_recommender import recommend_books

# create the window
root = tk.Tk()
root.title("Book Recommender")
root.configure(bg="#f5f5f5")


# user input function

def on_click():
    user_ID_text = entry.get()

    if not user_ID_text.isdigit():
        text.delete("1.0",tk.END)
        text.insert('1.0',"Invalid ID")
        entry.delete(0,tk.END)
        return

    user_id = int(user_ID_text)

    books = recommend_books(user_id)

    text.delete("1.0", tk.END)

    for i, book in enumerate(books, 1):
        text.insert(tk.END, f"{i}. {book}\n")

    

# create and show the widgets
entry = tk.Entry(root,width=30, font=("Helvetica", 12)) # text zone where the user will enter his ID
button = tk.Button(root,text="Get recommendations", command=on_click)
text = tk.Text(root,font=("Helvetica", 13)) # text zone that will show the recommended books

entry.pack(pady = 10)
button.pack()
text.pack()




root.mainloop()