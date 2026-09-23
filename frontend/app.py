import httpx
import streamlit as st

API_URL = "http://localhost:8000"
LOGO_PATH = "frontend/assets/dubizzle_logo.png"
ICON_PATH = "frontend/assets/dubizzle_icon.jpg"
ACCENT = "#ED1C24"
GREETING = "Hello! Welcome to dubizzle. What car are you looking for today?"

st.set_page_config(page_title="dubizzle | Car Shopping Assistant", page_icon=ICON_PATH, layout="wide")

st.markdown(
    f"""
    <style>
    .block-container {{ padding-top: 1.5rem; max-width: 1100px; }}
    .dubizzle-topbar {{
        height: 6px; margin: -1.5rem -1rem 1.5rem -1rem;
        background: {ACCENT};
    }}
    div[data-testid="stImage"] img {{ border-radius: 10px; }}
    </style>
    """,
    unsafe_allow_html=True,
)
st.markdown('<div class="dubizzle-topbar"></div>', unsafe_allow_html=True)

if "username" not in st.session_state:
    st.session_state.username = None
if "messages" not in st.session_state:
    st.session_state.messages = []
if "last_cars" not in st.session_state:
    st.session_state.last_cars = []
if "selected_car" not in st.session_state:
    st.session_state.selected_car = None

if st.session_state.username is None:
    _, center, _ = st.columns([1, 1, 1])
    with center:
        st.image(LOGO_PATH, width=220)
        st.markdown(
            "<p style='text-align:center; color:#555;'>AI car-shopping assistant</p>",
            unsafe_allow_html=True,
        )
        username_input = st.text_input("What's your name?", label_visibility="collapsed", placeholder="What's your name?")
        st.button("Continue", type="primary", use_container_width=True)
        if username_input:
            st.session_state.username = username_input
            st.session_state.messages = [("assistant", GREETING)]
            st.rerun()
    st.stop()

header_left, header_right = st.columns([3, 1])
with header_left:
    st.image(LOGO_PATH, width=140)
with header_right:
    st.markdown(
        f"<p style='text-align:right; padding-top: 1.2rem; color:#555;'>Signed in as <b>{st.session_state.username}</b></p>",
        unsafe_allow_html=True,
    )

with st.sidebar:
    st.image(LOGO_PATH, width=160)
    st.divider()
    if st.button("Start New Session", use_container_width=True):
        httpx.post(f"{API_URL}/new_session", json={"username": st.session_state.username})
        for key in ("username", "messages", "last_cars", "selected_car"):
            del st.session_state[key]
        st.rerun()

    st.divider()
    with st.expander("Dev Testing"):
        dev_view = st.radio(
            "Dev view", ["Off", "Cached Memory", "Long-Term Memory"],
            label_visibility="collapsed",
        )

if dev_view == "Cached Memory":
    st.subheader("Dev: Cached Memory (session_cache.json)")
    st.json(httpx.get(f"{API_URL}/dev/session_cache").json())
elif dev_view == "Long-Term Memory":
    st.subheader("Dev: Long-Term Memory (SQLite)")
    for table, rows in httpx.get(f"{API_URL}/dev/long_term_memory").json().items():
        st.markdown(f"**{table}** ({len(rows)} rows)")
        st.dataframe(rows, use_container_width=True)


def render_search_card(car: dict, col) -> None:
    # search-result cards: no favorite button here -- that only appears
    # once a car is actually selected, not before
    with col, st.container(border=True):
        if car.get("photo_url"):
            st.image(car["photo_url"], use_container_width=True)
        st.markdown(f"**{car.get('title', 'Untitled listing')}**")
        price = car.get("price_aed")
        st.write(f"AED {price:,.0f}" if price else "Price not mentioned")
        if st.button("Select", key=f"select_{car['listing_id']}", type="primary", use_container_width=True):
            response = httpx.post(f"{API_URL}/select_car", json={
                "username": st.session_state.username, "listing_id": car["listing_id"],
            })
            st.session_state.selected_car = response.json()
            st.rerun()


def render_selected_car(car: dict) -> None:
    st.subheader("Selected Car")
    with st.container(border=True):
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
            if st.button("Add to Favorites", key=f"fav_selected_{car['listing_id']}", type="primary"):
                httpx.post(f"{API_URL}/manage_favorite", json={
                    "username": st.session_state.username, "listing_id": car["listing_id"], "action": "add",
                })
                st.success("Added to favorites")


for role, content in st.session_state.messages:
    with st.chat_message(role, avatar=ICON_PATH if role == "assistant" else None):
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
            timeout=120,
        )
    data = response.json()
    st.session_state.messages.append(("assistant", data["reply"]))
    st.session_state.last_cars = data.get("cars", [])
    # always sync to the server's real selection -- the LLM can change it
    # mid-chat (e.g. while booking), not just via the Select button click
    st.session_state.selected_car = data.get("selected_car")
    st.rerun()
