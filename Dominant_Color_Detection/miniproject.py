import streamlit as st
import cv2
import numpy as np
from sklearn.cluster import KMeans
import imutils
from PIL import Image


def dominant_color_detection(uploaded_file, clusters):
    # Convert BytesIO to numpy array
    image_np = np.array(Image.open(uploaded_file))
    org_img = image_np.copy()
    img = cv2.cvtColor(image_np, cv2.COLOR_RGBA2BGR)  # Convert RGBA to BGR

    # Resize the image
    img = imutils.resize(img, height=200)

    # Flatten the image
    flat_img = np.reshape(img, (-1, 3))

    # Perform K-Means clustering
    kmeans = KMeans(n_clusters=clusters, random_state=0)
    kmeans.fit(flat_img)

    # Get dominant colors and percentages
    dominant_colors = np.array(kmeans.cluster_centers_, dtype='uint')
    percentages = (np.unique(kmeans.labels_, return_counts=True)[1]) / flat_img.shape[0]
    p_and_c = list(zip(percentages, dominant_colors))
    p_and_c = sorted(p_and_c, reverse=True)

    # Display the dominant colors
    st.subheader("Dominant Colors with Percentage")
    for i in range(clusters):
        st.image([np.full((50, 50, 3), p_and_c[i][1][::-1], dtype=np.uint8)],
                 caption=f"{round(p_and_c[i][0] * 100, 2)}%", width=50, use_column_width=False)
    
    st.markdown("<br>", unsafe_allow_html=True)

        
    # Display the proportions of colors in the image
    st.subheader("Visualizing Proportion of Colors")
    bar = np.ones((50, 500, 3), dtype='uint')
    start = 0
    i = 1
    for p, c in p_and_c:
        end = start + int(p * bar.shape[1])
        if i == clusters:
            bar[:, start:] = c[::-1]
        else:
            bar[:, start:end] = c[::-1]
        start = end
        i += 1

    st.image([bar], use_column_width=False)

    st.markdown("<br>", unsafe_allow_html=True)

    # Superimposed Image with Dominant Colors
    st.subheader("Image with Corresponding Dominant Colors")
    rows = 1000
    cols = int((org_img.shape[0] / org_img.shape[1]) * rows)
    img = cv2.resize(org_img, dsize=(rows, cols), interpolation=cv2.INTER_LINEAR)

    copy = img.copy()
    cv2.rectangle(copy, (rows // 2 - 650, cols // 2 - 190), (rows // 2 + 650, cols // 2 + 180), (255, 255, 255), -1)

    final = cv2.addWeighted(img, 0.1, copy, 0.9, 0)
    cv2.putText(final, 'Most Dominant Colors in the Image', (rows // 2 - 230, cols // 2 - 120),
                cv2.FONT_HERSHEY_DUPLEX, 0.8, (0, 0, 0), 1, cv2.LINE_AA)

    # Draw rectangles with dominant colors and index
    for i in range(clusters):
        start_x = int(final.shape[1] * i / clusters) + 20
        end_x = int(final.shape[1] * (i + 1) / clusters) - 20
        color = tuple(map(int, p_and_c[i][1][::-1]))  # Convert to tuple of integers
        pts = np.array([[start_x, final.shape[0] // 2 - 90],
                        [end_x, final.shape[0] // 2 - 90],
                        [end_x, final.shape[0] // 2 + 110],
                        [start_x, final.shape[0] // 2 + 110]], np.int32)
        pts = pts.reshape((-1, 1, 2))
        cv2.fillPoly(final, [pts], color)
        # Add index
        cv2.putText(final, str(i + 1), (start_x + 10, final.shape[0] // 2 + 60), cv2.FONT_HERSHEY_DUPLEX, 1,
                    (255, 255, 255), 1, cv2.LINE_AA)

    st.image(final, caption="Most Dominant Colors In The Image", use_column_width=True)
    st.set_option('deprecation.showPyplotGlobalUse', False)  # Disable the warning

    st.markdown("<hr> <br>", unsafe_allow_html=True)


    # Dropdown to select a dominant color
    selected_color_index = st.selectbox("Select a Dominant Color", [None] + list(range(1, clusters+1)))

    # Get the selected color
    selected_color = None if selected_color_index is None else p_and_c[selected_color_index - 1][1]

    # Filter the image based on the selected color
    filtered_image = filter_image(org_img, selected_color)

    # Display the filtered image
    st.subheader("Filtered Image")
    st.image(filtered_image, caption="Filtered Image", use_column_width=True)


def filter_image(original_image, selected_color, tolerance=50):
    if selected_color is None:
        return original_image

    hsv_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2HSV)
    lower_bound = np.array([selected_color[0] - tolerance, 50, 50])
    upper_bound = np.array([selected_color[0] + tolerance, 255, 255])
    mask = cv2.inRange(hsv_image, lower_bound, upper_bound)

    # Filter the image based on the selected color
    filtered_image = cv2.bitwise_and(original_image, original_image, mask=mask)

    # Set non-selected colors to white
    white_mask = cv2.bitwise_not(mask)
    filtered_image[white_mask > 0] = [255, 255, 255]

    return filtered_image



# Streamlit UI
title_html = f"""
    <div style="background-color:#FFC47E;padding:10px;border-radius:10px">
        <h1 style="color:white;text-align:center;">ColorHarmony</h1>
        <h4 style="color:#4F4A45;text-align:center;">Dominant Color Detection and Filtering</h4>
    </div>
"""
st.markdown(title_html, unsafe_allow_html=True)
st.markdown("<hr> <br>", unsafe_allow_html=True)
# Upload image through Streamlit
uploaded_file = st.file_uploader("Choose an image...", type="jpg")
st.markdown("<hr> <br>", unsafe_allow_html=True)

if uploaded_file is not None:
    # Get the number of clusters from the user
    clusters = st.slider("Select the number of clusters:", min_value=1, max_value=10, value=5)

    # Process and display the results
    dominant_color_detection(uploaded_file, clusters)
