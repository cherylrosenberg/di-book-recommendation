import streamlit as st

from movie_recommender import recommend_books

st.set_page_config(page_title="Book Recommender", page_icon="📚", layout="centered")

st.title("Book Recommender")
st.caption("Enter a BookCrossing user ID to get up to three suggested titles.")

user_id_text = st.text_input(
    "User ID",
    placeholder="Numeric ID only",
    help="Same IDs as in your dataset (digits only).",
)

if st.button("Get recommendations", type="primary"):
    text = user_id_text.strip()

    if not text:
        st.warning("Please enter a user ID.")
    elif not text.isdigit():
        st.warning("Invalid ID — use digits only.")
    else:
        user_id = int(text)
        with st.spinner("Finding recommendations…"):
            books = recommend_books(user_id)

        if books == ["User ID not found"]:
            st.warning("User ID not found.")
        elif not books:
            st.warning("No recommendations available for this user.")
        else:
            st.success("Recommendations")
            for i, book in enumerate(books, 1):
                st.write(f"{i}. {book}")

st.sidebar.markdown(
    "Run from the repository root so the data file loads:\n\n"
    "`streamlit run streamlit_app.py`"
)
