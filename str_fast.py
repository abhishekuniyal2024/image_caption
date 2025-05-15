import streamlit as st
import requests

# URL of your FastAPI endpoint (adjust if hosted elsewhere)
FASTAPI_URL = "http://127.0.0.1:8000/generate-caption"

def main():
    st.title("Image Caption Generator (FastAPI backend)")

    uploaded_image = st.file_uploader("Upload an image...", type=["jpg", "jpeg", "png"])

    if uploaded_image is not None:
        st.image(uploaded_image, caption="Uploaded Image", use_column_width=True)

        if st.button("Generate Caption"):
            # Prepare file for upload
            files = {"file": (uploaded_image.name, uploaded_image.getvalue(), uploaded_image.type)}

            # Send POST request to FastAPI
            with st.spinner("Generating caption..."):
                response = requests.post(FASTAPI_URL, files=files)

            if response.status_code == 200:
                caption = response.json().get("caption", "")
                st.success("Caption generated!")
                st.write(f"**Caption:** {caption}")
            else:
                st.error(f"Error: {response.text}")

if __name__ == "__main__":
    main()
