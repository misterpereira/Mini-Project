import streamlit as st
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import imutils
from PIL import Image

def draw_color_cube(color, percentage):
    cube_size = 50
    cube = np.ones((cube_size, cube_size, 3), dtype='uint') * color[::-1]
    plt.imshow(cube)
    plt.xticks([])
    plt.yticks([])
    plt.xlabel(f"{round(percentage * 100, 2)}%")

def dominant_color_detection(uploaded_file, clusters):
    # Convert BytesIO to numpy array
    image_np = np.array(Image.open(uploaded_file))
    org_img = image_np.copy()
    img = cv2.cvtColor(image_np, cv2.COLOR_RGBA2BGR)  # Convert RGBA to BGR
    # org_img = img.copy()

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
    st.subheader("Dominant Colors")
    for i in range(clusters):
        st.image([np.full((50, 50, 3), p_and_c[i][1][::-1], dtype=np.uint8)],
                 caption=f"{round(p_and_c[i][0] * 100, 2)}%", width=50, use_column_width=False)

    # Display the proportions of colors in the image
    st.subheader("Proportions of Colors")
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

    # Superimposed Image with Dominant Colors
    st.subheader("Superimposed Image with Dominant Colors")
    rows = 1000
    cols = int((org_img.shape[0] / org_img.shape[1]) * rows)
    img = cv2.resize(org_img, dsize=(rows, cols), interpolation=cv2.INTER_LINEAR)

    copy = img.copy()
    cv2.rectangle(copy, (rows // 2 - 650, cols // 2 - 190), (rows // 2 + 650, cols // 2 + 180), (255, 255, 255), -1)

    # st.image(org_img, caption="Most Dominant Colors In The Image", use_column_width=True)

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

# Streamlit UI
st.title("Dominant Color Detection App")

# Upload image through Streamlit
uploaded_file = st.file_uploader("Choose an image...", type="jpg")

if uploaded_file is not None:
    # Get the number of clusters from the user
    clusters = st.slider("Select the number of clusters:", min_value=1, max_value=10, value=5)

    # Process and display the results
    dominant_color_detection(uploaded_file, clusters)
