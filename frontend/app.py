import httpx
import streamlit as st

API_URL = "http://localhost:8000"

st.set_page_config(page_title="Car Shopping Assistant", layout="wide")
st.title("Car Shopping Assistant")

if "username" not in st.session_state:
    st.session_state.username = None
if "messages" not in st.session_state:
    st.session_state.messages = []
if "last_cars" not in st.session_state:
    st.session_state.last_cars = []

if st.session_state.username is None:
    username_input = st.text_input("What's your name?")
    if username_input:
        st.session_state.username = username_input
        st.rerun()
    st.stop()

st.caption(f"Signed in as {st.session_state.username}")


def render_car_card(car: dict, col) -> None:
    with col:
        if car.get("photo_url"):
            st.image(car["photo_url"], use_container_width=True)
        st.markdown(f"**{car.get('title', 'Untitled listing')}**")
        price = car.get("price_aed")
        st.write(f"AED {price:,.0f}" if price else "Price not mentioned")
        c1, c2 = st.columns(2)
        if c1.button("Select", key=f"select_{car['listing_id']}"):
            httpx.post(f"{API_URL}/select_car", json={"username": st.session_state.username, "listing_id": car["listing_id"]})
            st.success(f"Selected {car['make']} {car['model']}")
        if c2.button("Favorite", key=f"fav_{car['listing_id']}"):
            httpx.post(f"{API_URL}/manage_favorite", json={
                "username": st.session_state.username, "listing_id": car["listing_id"], "action": "add",
            })
            st.success("Added to favorites")


for role, content in st.session_state.messages:
    with st.chat_message(role):
        st.write(content)

if st.session_state.last_cars:
    st.subheader("Cars")
    cols = st.columns(3)
    for i, car in enumerate(st.session_state.last_cars):
        render_car_card(car, cols[i % 3])

user_message = st.chat_input("Ask about cars...")
if user_message:
    st.session_state.messages.append(("user", user_message))
    with st.spinner("Thinking..."):
        response = httpx.post(
            f"{API_URL}/chat",
            json={"username": st.session_state.username, "message": user_message},
            timeout=60,
        )
    data = response.json()
    st.session_state.messages.append(("assistant", data["reply"]))
    st.session_state.last_cars = data.get("cars", [])
    st.rerun()
