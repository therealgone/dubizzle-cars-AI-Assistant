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
if "selected_car" not in st.session_state:
    st.session_state.selected_car = None

if st.session_state.username is None:
    username_input = st.text_input("What's your name?")
    if username_input:
        st.session_state.username = username_input
        st.rerun()
    st.stop()

st.caption(f"Signed in as {st.session_state.username}")

with st.sidebar:
    if st.button("Start New Session"):
        httpx.post(f"{API_URL}/new_session", json={"username": st.session_state.username})
        for key in ("username", "messages", "last_cars", "selected_car"):
            del st.session_state[key]
        st.rerun()


def render_search_card(car: dict, col) -> None:
    # search-result cards: no favorite button here -- that only appears
    # once a car is actually selected, not before
    with col:
        if car.get("photo_url"):
            st.image(car["photo_url"], use_container_width=True)
        st.markdown(f"**{car.get('title', 'Untitled listing')}**")
        price = car.get("price_aed")
        st.write(f"AED {price:,.0f}" if price else "Price not mentioned")
        if st.button("Select", key=f"select_{car['listing_id']}"):
            response = httpx.post(f"{API_URL}/select_car", json={
                "username": st.session_state.username, "listing_id": car["listing_id"],
            })
            st.session_state.selected_car = response.json()
            st.rerun()


def render_selected_car(car: dict) -> None:
    st.subheader("Selected Car")
    col1, col2 = st.columns([1, 2])
    with col1:
        if car.get("photo_url"):
            st.image(car["photo_url"], use_container_width=True)
    with col2:
        st.markdown(f"### {str(car.get('make', '')).title()} {str(car.get('model', '')).title()} {car.get('trim', '')}")
        st.write(f"**Year:** {car.get('year')}")
        st.write(f"**Body type:** {car.get('body_type')}")
        st.write(f"**Color:** {car.get('color')}")
        price = car.get("price_aed")
        st.write(f"**Price:** AED {price:,.0f}" if price else "**Price:** Not mentioned")
        st.caption(car.get("description", ""))
        if st.button("Add to Favorites", key=f"fav_selected_{car['listing_id']}"):
            httpx.post(f"{API_URL}/manage_favorite", json={
                "username": st.session_state.username, "listing_id": car["listing_id"], "action": "add",
            })
            st.success("Added to favorites")


for role, content in st.session_state.messages:
    with st.chat_message(role):
        st.write(content)

if st.session_state.selected_car and "error" not in st.session_state.selected_car:
    render_selected_car(st.session_state.selected_car)

if st.session_state.last_cars:
    st.subheader("Cars")
    cols = st.columns(3)
    for i, car in enumerate(st.session_state.last_cars):
        render_search_card(car, cols[i % 3])

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
    # always sync to the server's real selection -- the LLM can change it
    # mid-chat (e.g. while booking), not just via the Select button click
    st.session_state.selected_car = data.get("selected_car")
    st.rerun()
