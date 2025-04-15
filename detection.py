import json
import os
import time
import uuid
from datetime import datetime
from pathlib import Path

import cv2
import helper
import numpy as np
import PIL
import settings
import streamlit as st

# Import the updated modules
from modules.gps_utils import get_gps_location, save_location_data
from modules.image_uploader import authenticate_drive, upload_image
from modules.processing import format_detection_results, non_max_suppression
from modules.visualizations import draw_bounding_boxes, run_live_detection
from PIL import Image
from streamlit_extras.stylable_container import stylable_container


def load_disease_data():
    """Load disease information from the JSON file."""
    try:
        with open("dataset/diseases.json", "r") as f:
            disease_data = json.load(f)

        # Create a lookup dictionary by lowercase disease name
        disease_lookup = {}
        for disease in disease_data:
            disease_lookup[disease["title"].lower()] = disease

        return disease_lookup
    except Exception as e:
        st.warning(f"Could not load disease information: {e}")
        return {}


def get_disease_details(disease_name, disease_lookup):
    """Get details for a specific disease."""
    # Map model labels to JSON titles if necessary
    name_mapping = {
        "abiotic disorder": "abiotic disorder",
        "algal growth": "algae growth",
        "cercospora": "cercospora",
        "late stage rust": "leaf rust",
        "rust": "leaf rust",
        "sooty mold": "sooty mold",
    }

    search_name = name_mapping.get(disease_name.lower(), disease_name.lower())
    return disease_lookup.get(search_name, None)


def main(theme_colors):
    # Get theme colors
    primary_color = theme_colors["primaryColor"]
    background_color = theme_colors["backgroundColor"]
    secondary_background_color = theme_colors["secondaryBackgroundColor"]
    text_color = theme_colors["textColor"]

    # Apply theme to Streamlit
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-color: {background_color};
            color: {text_color};
        }}
        .stSelectbox, .stSelectbox > div > div > div {{
            background-color: {secondary_background_color};
            color: {text_color};
        }}
        .stButton button {{
            background-color: {primary_color};
            color: white;
            border-radius: 8px;
            padding: 0.5rem 1.5rem;
            font-weight: 600;
            border: none;
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
            transition: all 0.3s ease;
        }}
        .stButton button:hover {{
            transform: translateY(-2px);
            box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.1), 0 4px 6px -2px rgba(0, 0, 0, 0.05);
        }}
        div[data-testid="stExpander"] {{
            border-radius: 8px;
            overflow: hidden;
            box-shadow: 0 1px 3px 0 rgba(0, 0, 0, 0.1), 0 1px 2px 0 rgba(0, 0, 0, 0.06);
            margin-bottom: 1rem;
        }}
        div[data-testid="stExpander"] details summary p {{
            font-weight: 600;
        }}
        div[data-testid="stExpander"] details div.streamlit-expanderContent {{
            border-top: 1px solid {primary_color}30;
            padding-top: 1rem;
        }}
        div[data-testid="stVerticalBlock"] div[data-testid="stHorizontalBlock"] {{
            gap: 1rem;
        }}
        </style>
    """,
        unsafe_allow_html=True,
    )

    # Define global color mappings for classes
    cleaf_colors = {
        0: (0, 255, 0),  # Green for 'arabica'
        1: (0, 255, 255),  # Aqua for 'liberica'
        2: (0, 0, 255),  # Blue for 'robusta'
    }

    cdisease_colors = {
        0: (0, 128, 128),  # Teal for 'abiotic disorder'
        1: (255, 165, 0),  # Orange for 'algal growth'
        2: (255, 0, 0),  # Red for 'cercospora'
        3: (128, 0, 128),  # Purple for 'late stage rust'
        4: (150, 75, 0),  # Magenta for 'rust'
        5: (255, 255, 0),  # Yellow for 'sooty mold'
    }

    # Sidebar
    with st.sidebar:
        st.markdown(
            f"""
        <div style="text-align: center; margin-bottom: 16px;">
            <h2 style="color: {primary_color}; font-weight: 600; margin-bottom: 2px;">MODEL SETTINGS</h2>
            <p style="font-size: 0.9rem; opacity: 0.8;">Configure detection parameters</p>
        </div>
        """,
            unsafe_allow_html=True,
        )

        with st.container(border=False):
            st.markdown(
                f"""
            <div style="margin-bottom: 5px;">
                <span style="font-weight: 600; color: {primary_color};">Select Detection Model</span>
            </div>
            """,
                unsafe_allow_html=True,
            )

            detection_model_choice = st.selectbox(
                "Select Detection Model",
                ("Disease", "Leaf", "Both Models"),
                index=0,
                label_visibility="collapsed",
            )

        st.divider()

        with st.container(border=False):
            adv_opt = st.toggle(
                "Advanced Options",
                help="Configure confidence threshold and overlap settings",
            )

            if adv_opt:
                confidence = (
                    float(
                        st.slider(
                            "Confidence Threshold",
                            25,
                            100,
                            40,
                            help="Minimum confidence level required for detections",
                        )
                    )
                    / 100
                )
                overlap_threshold = (
                    float(
                        st.slider(
                            "Overlap Threshold",
                            0,
                            100,
                            30,
                            help="Controls how objects can overlap",
                        )
                    )
                    / 100
                )
            else:
                confidence = 0.4
                overlap_threshold = 0.3

        st.divider()

        with st.container(border=False):
            st.markdown(
                f"""
            <div style="margin-bottom: 5px;">
                <span style="font-weight: 600; color: {primary_color};">Upload & Save Options</span>
            </div>
            """,
                unsafe_allow_html=True,
            )

            # ✅ GDrive checkbox
            save_to_drive = st.checkbox(
                "📤 Save samples to improve the model",
                help="Uploads anonymous detection data to improve future model versions",
            )

            source_img = st.file_uploader(
                "Choose an image...", type=("jpg", "jpeg", "png", "bmp", "webp")
            )

        if source_img:
            st.divider()
            detect_btn = st.button("🔍 Detect Objects", type="primary")
        else:
            st.divider()
            st.button(
                "🔍 Detect Objects",
                type="primary",
                disabled=True,
                help="Please upload an image first",
            )

    # Load disease information
    disease_lookup = load_disease_data()

    # ✅ Authenticate Drive only once
    @st.cache_resource
    def get_drive():
        return authenticate_drive()

    drive = get_drive()
    PARENT_FOLDER_ID = (
        "1OgdV5CRT61ujv1uW1SSgnesnG59bT5ss"  # ✅ Your actual GDrive folder
    )

    with stylable_container(
        key="container_with_border",
        css_styles=f"""
                {{
                    background-color: {secondary_background_color};
                    border-radius: 12px;
                    padding: 1.5rem;
                    max-width: 1200px;
                    margin: auto;
                    box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.1), 0 4px 6px -2px rgba(0, 0, 0, 0.05);
                }}
                """,
    ):
        col1, col2 = st.columns(2)

        with col1:
            with st.container(border=True):
                image_placeholder = st.empty()

                try:
                    if source_img is None:
                        default_image_path = str(settings.DEFAULT_DETECT_IMAGE)
                        default_image = PIL.Image.open(default_image_path)
                        image_placeholder.image(
                            default_image_path,
                            caption="Sample Image: Objects Detected",
                            width=500,
                        )
                    else:
                        uploaded_image = PIL.Image.open(source_img)
                        image_placeholder.image(
                            source_img, caption="Uploaded Image", width=500
                        )
                except Exception as ex:
                    st.error("Error occurred while opening the image.")
                    st.error(ex)

        with col2:
            with stylable_container(
                key="container_with_border1",
                css_styles=f"""
                        {{
                            background-color: {background_color};
                            border-radius: 10px;
                            max-width: 694px;
                            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
                        }}
                        """,
            ):
                col2_placeholder = st.empty()

                with col2_placeholder.container():
                    st.markdown(
                        f"<div style='border-radius: 10px 10px 0px 0px; border-bottom: 2px solid {primary_color}; padding: 16px; background-color: {secondary_background_color};'><h3 style='font-weight: 600; font-size: 1.35rem; color: {primary_color}; margin: 0;'>INSTRUCTIONS</h3></div>",
                        unsafe_allow_html=True,
                    )
                    st.markdown(
                        f"""
                            <div style='padding: 20px; font-size: 1rem; color: {text_color};'>
                                <ol style='padding-left: 20px; margin-top: 0;'>
                                    <li>Open the sidebar to start configuring</li>
                                    <li>Upload a valid image file (jpeg, jpg, webp, png)</li>
                                    <li>Click "<strong>Detect Objects</strong>" to analyze</li>
                                    <li>Review the results in the popup window</li>
                                </ol>
                                <p style='margin-top: 16px;'>
                                    <strong>🌿 Best practices:</strong> Ensure that the photo clearly shows a coffee leaf. 
                                    Avoid bluriness and make sure the leaf is the main focus of the image.
                                </p>
                            </div>
                        """,
                        unsafe_allow_html=True,
                    )
                    st.markdown(
                        f"""
                            <div style='background-color: {primary_color}15; border-radius: 0 0 10px 10px; border-top: 1px solid {primary_color}30; font-size: 0.9rem; color: {primary_color}; padding: 12px 20px;'>
                                <strong>Note:</strong> Our model is currently optimized to detect diseases only in coffee leaves.
                            </div>
                        """,
                        unsafe_allow_html=True,
                    )

    if source_img:
        uploaded_image = PIL.Image.open(source_img)
        image_placeholder.image(uploaded_image, caption="Uploaded Image", width=500)

        # Extract GPS metadata
        gps_data = get_gps_location(source_img)

        if gps_data:
            with st.expander("Image Location Data"):
                st.write(f"Latitude: {gps_data['latitude']:.6f}°")
                st.write(f"Longitude: {gps_data['longitude']:.6f}°")
                if gps_data["altitude"]:
                    st.write(f"Altitude: {float(gps_data['altitude']):.1f}m")
                st.map({"lat": [gps_data["latitude"]], "lon": [gps_data["longitude"]]})

        # Move the "if st.sidebar.button" check to here
        if detect_btn:
            # Load model based on selection
            try:
                if detection_model_choice == "Disease":
                    model_path = Path(settings.DISEASE_DETECTION_MODEL)
                    model = helper.load_model(model_path)
                    colors = cdisease_colors
                elif detection_model_choice == "Leaf":
                    model_path = Path(settings.LEAF_DETECTION_MODEL)
                    model = helper.load_model(model_path)
                    colors = cleaf_colors
                else:
                    model_disease = helper.load_model(
                        Path(settings.DISEASE_DETECTION_MODEL)
                    )
                    model_leaf = helper.load_model(Path(settings.LEAF_DETECTION_MODEL))
            except Exception as ex:
                st.error(f"Error loading model: {ex}")
                return

            if detection_model_choice == "Both Models":
                # Run both models
                @st.dialog("Results")
                def both_models():
                    # Predict using both disease and leaf models
                    res_disease = model_disease.predict(uploaded_image, conf=confidence)
                    res_leaf = model_leaf.predict(uploaded_image, conf=confidence)

                    # Apply non-max suppression
                    disease_boxes = non_max_suppression(
                        res_disease[0].boxes, overlap_threshold
                    )
                    leaf_boxes = non_max_suppression(
                        res_leaf[0].boxes, overlap_threshold
                    )

                    # Prepare result image
                    result_image = np.array(uploaded_image)
                    result_image = draw_bounding_boxes(
                        result_image,
                        disease_boxes,
                        res_disease[0].names,
                        cdisease_colors,
                    )
                    result_image = draw_bounding_boxes(
                        result_image, leaf_boxes, res_leaf[0].names, cleaf_colors
                    )

                    # Display the image
                    with st.container(border=True):
                        st.image(result_image, caption="Detected Image", width=450)

                    saved_any_detections = False  # Track if anything was saved
                    uploaded = False
                    image_placeholder.image(
                        result_image, caption="Detected Image", width=450
                    )

                    # Display overall disease status in a prominent way
                    if len(disease_boxes) == 0:
                        st.markdown(
                            f"""
                            <div style="display: flex; align-items: center; gap: 10px; background-color: #d1e7dd; color: #0f5132; padding: 12px; border-radius: 8px; margin: 16px 0;">
                                <span style="font-size: 1.5rem;">✅</span>
                                <div>
                                    <h3 style="margin: 0; font-weight: 600; font-size: 1.1rem;">HEALTHY LEAF</h3>
                                    <p style="margin: 4px 0 0 0; font-size: 0.9rem;">No diseases detected in this sample</p>
                                </div>
                            </div>
                        """,
                            unsafe_allow_html=True,
                        )
                    else:
                        st.markdown(
                            f"""
                            <div style="display: flex; align-items: center; gap: 10px; background-color: #fff3cd; color: #664d03; padding: 12px; border-radius: 8px; margin: 16px 0;">
                                <span style="font-size: 1.5rem;">⚠️</span>
                                <div>
                                    <h3 style="margin: 0; font-weight: 600; font-size: 1.1rem;">DISEASES DETECTED</h3>
                                    <p style="margin: 4px 0 0 0; font-size: 0.9rem;">{len(disease_boxes)} issue(s) found in this sample</p>
                                </div>
                            </div>
                        """,
                            unsafe_allow_html=True,
                        )

                    # Process disease detections
                    for box in disease_boxes:
                        class_id = int(box.cls[0])
                        confidence_score = round(float(box.conf[0]) * 100, 1)
                        disease_name = res_disease[0].names[class_id]

                        if confidence_score > 50:
                            with st.spinner("Adding disease to the database..."):
                                save_location_data(
                                    source_img, disease_name, confidence_score, gps_data
                                )
                            saved_any_detections = True

                            # Upload to drive only once per image
                            if save_to_drive and not uploaded:
                                temp_path = f"temp_{uuid.uuid4().hex}.jpg"
                                uploaded_image.save(temp_path)
                                try:
                                    result = upload_image(
                                        temp_path, disease_name, drive, PARENT_FOLDER_ID
                                    )
                                    st.toast(result)
                                except Exception as e:
                                    st.error(f"Drive upload failed: {e}")
                                os.remove(temp_path)
                                uploaded = True

                            # Display disease details in an expander
                            disease_details = get_disease_details(
                                disease_name, disease_lookup
                            )
                            if disease_details:
                                with st.expander(
                                    f"🔍 {disease_name.upper()} Details & Treatment"
                                ):
                                    cols = st.columns([1, 2])

                                    with cols[0]:
                                        img_path = disease_details.get("image", "")
                                        if img_path:
                                            try:
                                                st.image(img_path, width=200)
                                            except:
                                                st.info("Image preview not available")

                                    with cols[1]:
                                        st.markdown(
                                            f"<h3 style='color: {primary_color}; margin-top: 0; font-size: 1.2rem;'>{disease_details.get('title')}</h3>",
                                            unsafe_allow_html=True,
                                        )
                                        if "name2" in disease_details:
                                            st.markdown(
                                                f"<p style='font-style: italic; margin: 8px 0; font-size: 0.9rem;'>Scientific name: <strong>{disease_details.get('name2')}</strong></p>",
                                                unsafe_allow_html=True,
                                            )

                                            st.markdown(
                                                f"<div style='background-color: {primary_color}10; border-left: 3px solid {primary_color}; padding: 10px; margin: 12px 0; font-size: 0.95rem;'>{disease_details.get('description')}</div>",
                                                unsafe_allow_html=True,
                                            )

                                        if "prevention" in disease_details:
                                            st.markdown(
                                                f"<h4 style='color: {primary_color}; border-bottom: 1px solid {primary_color}20; padding-bottom: 8px;'>Prevention Methods</h4>",
                                                unsafe_allow_html=True,
                                            )
                                            for i, method in enumerate(
                                                disease_details.get("prevention")
                                            ):
                                                st.markdown(
                                                    f"""
                                                    <div style='display: flex; gap: 10px; margin-bottom: 8px; align-items: flex-start;'>
                                                        <div style='background-color: {primary_color}; color: white; border-radius: 50%; width: 24px; height: 24px; display: flex; align-items: center; justify-content: center; flex-shrink: 0;'>{i+1}</div>
                                                        <div>{method}</div>
                                                    </div>
                                                """,
                                                    unsafe_allow_html=True,
                                                )

                                        if "solution" in disease_details:
                                            st.markdown(
                                                f"<h4 style='color: {primary_color}; border-bottom: 1px solid {primary_color}20; padding-bottom: 8px; margin-top: 20px;'>Treatment Solutions</h4>",
                                                unsafe_allow_html=True,
                                            )
                                            for solution in disease_details.get(
                                                "solution"
                                            ):
                                                st.markdown(
                                                    f"""
                                                    <div style='background-color: {primary_color}15; margin-bottom: 8px; padding: 10px; border-radius: 6px;'>
                                                        <div style='display: flex; align-items: center;'>
                                                            <span style='color: {primary_color}; margin-right: 8px;'>✓</span>
                                                            {solution}
                                                        </div>
                                                    </div>
                                                """,
                                                    unsafe_allow_html=True,
                                                )

                    # Process leaf detections
                    for box in leaf_boxes:
                        class_id = int(box.cls[0])
                        confidence_score = round(float(box.conf[0]) * 100, 1)
                        leaf_name = res_leaf[0].names[class_id]

                        if confidence_score > 50:
                            # Skip coffee variety detection
                            if leaf_name.lower() in ["arabica", "liberica", "robusta"]:
                                continue

                            save_location_data(
                                source_img, leaf_name, confidence_score, gps_data
                            )
                            saved_any_detections = True

                            # Upload to drive only once per image
                            if save_to_drive and not uploaded:
                                temp_path = f"temp_{uuid.uuid4().hex}.jpg"
                                uploaded_image.save(temp_path)
                                try:
                                    result = upload_image(
                                        temp_path, leaf_name, drive, PARENT_FOLDER_ID
                                    )
                                    st.toast(result)
                                except Exception as e:
                                    st.error(f"Drive upload failed: {e}")
                                os.remove(temp_path)
                                uploaded = True

                    # Log disease detections
                    results_col = st.columns(2)
                    with results_col[0]:
                        if len(disease_boxes) > 0:
                            with st.popover("See disease results"):
                                for box in disease_boxes:
                                    class_id = int(box.cls[0])
                                    confidence_score = round(
                                        float(box.conf[0]) * 100, 1
                                    )
                                    disease_name = res_disease[0].names[class_id]
                                    st.write(
                                        f"- **{disease_name}**: {confidence_score}% confidence"
                                    )
                        else:
                            with st.popover("See disease results"):
                                st.write("- **Healthy Leaf**: No diseases detected")

                    # Log leaf detections
                    with results_col[1]:
                        with st.popover("See leaf results"):
                            for box in leaf_boxes:
                                class_id = int(box.cls[0])
                                confidence_score = round(float(box.conf[0]) * 100, 1)
                                leaf_name = res_leaf[0].names[class_id]
                                st.write(
                                    f"- **{leaf_name}**: {confidence_score}% confidence"
                                )

                    # Show success message if any detections were saved
                    if saved_any_detections:
                        st.success("✅ Data saved successfully!")

                both_models()  # Call the function

            else:
                # Run selected model
                @st.dialog("Results")
                def run_model():
                    # Predict using the selected model
                    if detection_model_choice == "Leaf":
                        res = model_leaf.predict(uploaded_image, conf=confidence)
                    else:
                        res = model.predict(uploaded_image, conf=confidence)

                    # Apply non-max suppression
                    boxes = non_max_suppression(res[0].boxes, overlap_threshold)
                    labels = res[0].names

                    # Draw bounding boxes on the image
                    result_image = draw_bounding_boxes(
                        uploaded_image, boxes, labels, colors
                    )

                    # Display the image
                    with st.container(border=True):
                        st.image(result_image, caption="Detected Image", width=450)

                    saved_any_detections = False
                    uploaded = False
                    detection_results = []

                    image_placeholder.image(
                        result_image, caption="Detected Image", width=450
                    )

                    # Check for disease model and show health status
                    if detection_model_choice == "Disease":
                        if len(boxes) == 0:
                            st.markdown(
                                f"""
                                <div style="display: flex; align-items: center; gap: 10px; background-color: #d1e7dd; color: #0f5132; padding: 12px; border-radius: 8px; margin: 16px 0;">
                                    <span style="font-size: 1.5rem;">✅</span>
                                    <div>
                                        <h3 style="margin: 0; font-weight: 600; font-size: 1.1rem;">HEALTHY LEAF</h3>
                                        <p style="margin: 4px 0 0 0; font-size: 0.9rem;">No diseases detected in this sample</p>
                                    </div>
                                </div>
                            """,
                                unsafe_allow_html=True,
                            )
                            detection_results.append(
                                "- **Healthy Leaf**: No diseases detected"
                            )
                        else:
                            st.markdown(
                                f"""
                                <div style="display: flex; align-items: center; gap: 10px; background-color: #fff3cd; color: #664d03; padding: 12px; border-radius: 8px; margin: 16px 0;">
                                    <span style="font-size: 1.5rem;">⚠️</span>
                                    <div>
                                        <h3 style="margin: 0; font-weight: 600; font-size: 1.1rem;">DISEASES DETECTED</h3>
                                        <p style="margin: 4px 0 0 0; font-size: 0.9rem;">{len(boxes)} issue(s) found in this sample</p>
                                    </div>
                                </div>
                            """,
                                unsafe_allow_html=True,
                            )

                    for box in boxes:
                        class_id = int(box.cls[0])
                        conf_score = round(float(box.conf[0]) * 100, 1)
                        detection_name = labels[class_id]
                        # Store results for the popover
                        detection_results.append(
                            f"- **{detection_name}**: {conf_score}% confidence"
                        )

                        # Skip coffee varieties
                        if detection_name.lower() in ["arabica", "liberica", "robusta"]:
                            continue

                        if conf_score > 50:
                            with st.spinner("Adding disease to the database..."):
                                save_location_data(
                                    source_img, detection_name, conf_score, gps_data
                                )
                            saved_any_detections = True

                            # Upload to drive only once per image
                            if save_to_drive and not uploaded:
                                temp_path = f"temp_{uuid.uuid4().hex}.jpg"
                                uploaded_image.save(temp_path)
                                try:
                                    result = upload_image(
                                        temp_path,
                                        detection_name,
                                        drive,
                                        PARENT_FOLDER_ID,
                                    )
                                    st.toast(result)
                                except Exception as e:
                                    st.error(f"Drive upload failed: {e}")
                                os.remove(temp_path)
                                uploaded = True

                            # Display disease details in an expander (for Disease model)
                            if detection_model_choice == "Disease":
                                disease_details = get_disease_details(
                                    detection_name, disease_lookup
                                )
                                if disease_details:
                                    with st.expander(
                                        f"🔍 {detection_name.upper()} Details & Treatment"
                                    ):
                                        cols = st.columns([1, 2])

                                        with cols[0]:
                                            img_path = disease_details.get("image", "")
                                            if img_path:
                                                try:
                                                    st.image(img_path, width=200)
                                                except:
                                                    st.info(
                                                        "Image preview not available"
                                                    )

                                        with cols[1]:
                                            st.markdown(
                                                f"<h3 style='color: {primary_color}; margin-top: 0; font-size: 1.2rem;'>{disease_details.get('title')}</h3>",
                                                unsafe_allow_html=True,
                                            )
                                            if "name2" in disease_details:
                                                st.markdown(
                                                    f"<p style='font-style: italic; margin: 8px 0; font-size: 0.9rem;'>Scientific name: <strong>{disease_details.get('name2')}</strong></p>",
                                                    unsafe_allow_html=True,
                                                )

                                            st.markdown(
                                                f"<div style='background-color: {primary_color}10; border-left: 3px solid {primary_color}; padding: 10px; margin: 12px 0; font-size: 0.95rem;'>{disease_details.get('description')}</div>",
                                                unsafe_allow_html=True,
                                            )

                                        if "prevention" in disease_details:
                                            st.markdown(
                                                f"<h4 style='color: {primary_color}; border-bottom: 1px solid {primary_color}20; padding-bottom: 8px;'>Prevention Methods</h4>",
                                                unsafe_allow_html=True,
                                            )
                                            for i, method in enumerate(
                                                disease_details.get("prevention")
                                            ):
                                                st.markdown(
                                                    f"""
                                                    <div style='display: flex; gap: 10px; margin-bottom: 8px; align-items: flex-start;'>
                                                        <div style='background-color: {primary_color}; color: white; border-radius: 50%; width: 24px; height: 24px; display: flex; align-items: center; justify-content: center; flex-shrink: 0;'>{i+1}</div>
                                                        <div>{method}</div>
                                                    </div>
                                                """,
                                                    unsafe_allow_html=True,
                                                )

                                        if "solution" in disease_details:
                                            st.markdown(
                                                f"<h4 style='color: {primary_color}; border-bottom: 1px solid {primary_color}20; padding-bottom: 8px; margin-top: 20px;'>Treatment Solutions</h4>",
                                                unsafe_allow_html=True,
                                            )
                                            for solution in disease_details.get(
                                                "solution"
                                            ):
                                                st.markdown(
                                                    f"""
                                                    <div style='background-color: {primary_color}15; margin-bottom: 8px; padding: 10px; border-radius: 6px;'>
                                                        <div style='display: flex; align-items: center;'>
                                                            <span style='color: {primary_color}; margin-right: 8px;'>✓</span>
                                                            {solution}
                                                        </div>
                                                    </div>
                                                """,
                                                    unsafe_allow_html=True,
                                                )

                    if detection_results:
                        with st.popover("See advanced results"):
                            st.write("\n".join(detection_results))
                    else:
                        with st.popover("See advanced results"):
                            if detection_model_choice == "Disease":
                                st.write("- **Healthy Leaf**: No diseases detected")
                            else:
                                st.write("- No detections found")
                    # Show success message if any detections were saved
                    if saved_any_detections:
                        st.success("✅ Data saved successfully!")

                # Always call the modal dialog
                run_model()


if __name__ == "__main__":
    default_theme = {
        "primaryColor": "#FF7E00",
        "backgroundColor": "white",
        "secondaryBackgroundColor": "#fafafa",
        "textColor": "black",
    }
    main(default_theme)
