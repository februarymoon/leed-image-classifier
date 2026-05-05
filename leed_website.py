import streamlit as st
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
import torch

st.set_page_config(
    page_title="LEED Image Classifier",
    page_icon="🌿",
    layout="wide"
)

st.markdown("""
<style>
    .main {
        background-color: #f7f9f6;
    }

    .title {
        font-size: 42px;
        font-weight: 800;
        color: #2f4f2f;
        margin-bottom: 5px;
    }

    .subtitle {
        font-size: 18px;
        color: #5f6f5f;
        margin-bottom: 30px;
    }

    .result-box {
        padding: 25px;
        border-radius: 18px;
        background-color: #ffffff;
        box-shadow: 0px 4px 15px rgba(0,0,0,0.08);
        margin-top: 20px;
    }

    .category {
        font-size: 28px;
        font-weight: 700;
        color: #2f4f2f;
    }

    .note {
        font-size: 14px;
        color: #777;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_model():
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    return model, processor


model, processor = load_model()


leed_categories = {
    "Energy and Atmosphere": [
        "solar panels on a building roof",
        "photovoltaic panels integrated into a building facade",
        "solar canopy shading a parking area",
        "renewable energy system connected to a building",
        "deep overhangs shading windows",
        "external louvers blocking direct sunlight",
        "perforated facade screen used for shading",
        "double skin facade reducing heat gain",
        "brise soleil shading a building facade",
        "shaded courtyard reducing cooling demand",
        "green roof reducing heat gain",
        "white reflective roof reducing heat absorption",
        "thermal mass walls used for passive cooling",
        "thick exterior walls reducing heat gain",
        "small recessed windows in a thick wall",
        "high performance glass facade",
        "double glazed windows",
        "insulated building envelope",
        "LED lighting fixtures in a building",
        "occupancy sensor controlling lights",
        "daylight sensor for artificial lighting control",
        "efficient HVAC equipment",
        "mechanical room with efficient cooling equipment",
        "VRF air conditioning system",
        "high efficiency chiller",
        "heat recovery ventilation unit",
        "air handling unit with energy recovery",
        "building energy meter",
        "energy monitoring display screen",
        "building management system screen",
        "smart thermostat",
        "energy control panel",
        "building designed for low energy consumption",
        "passive design strategy visible on building",
        "sustainable building facade reducing energy demand"
    ],

    "Indoor Environmental Quality": [
        "daylit interior space",
        "large windows bringing daylight into a room",
        "skylight bringing natural light indoors",
        "clerestory windows providing daylight",
        "atrium bringing daylight into the building",
        "bright interior with natural daylight",
        "light shelves reflecting daylight into a room",
        "controlled daylight with blinds or shading",
        "large windows with outdoor views",
        "interior workspace with view to landscape",
        "classroom with view to outside",
        "seating area with visual connection to nature",
        "interior space overlooking courtyard",
        "operable windows for natural ventilation",
        "open windows allowing fresh air into a room",
        "cross ventilation through opposite windows",
        "ventilation openings in a wall",
        "ceiling fans improving air movement",
        "natural ventilation strategy in an interior space",
        "indoor plants improving indoor air quality",
        "green wall inside a building",
        "fresh air diffuser in ceiling",
        "CO2 sensor in an interior space",
        "air quality monitor in a room",
        "healthy indoor environment",
        "thermostat for occupant temperature control",
        "individual thermal comfort control",
        "shading blinds controlling heat and glare",
        "comfortable indoor workspace",
        "interior space designed for thermal comfort",
        "adjustable task lighting",
        "individual lighting control",
        "glare controlled interior lighting",
        "soft interior lighting",
        "well lit interior workspace",
        "acoustic ceiling panels",
        "sound absorbing wall panels",
        "acoustic treatment in an auditorium",
        "carpet or soft surfaces reducing noise",
        "quiet study space with acoustic treatment",
        "wood acoustic panels in an interior",
        "perforated acoustic ceiling",
        "low emitting interior materials",
        "healthy comfortable indoor space"
    ],

    "Location and Transportation": [
        "bus stop near a building",
        "metro station entrance near a building",
        "tram stop near a building",
        "train station near a building",
        "public transportation stop near a building entrance",
        "transit shelter on a walkable street",
        "building located near public transport",
        "pedestrian walkway connected to a building",
        "wide sidewalk in front of a building",
        "safe pedestrian crossing near a building",
        "zebra crossing leading to a building entrance",
        "walkable street with active building frontage",
        "shaded pedestrian path",
        "pedestrian plaza in front of a building",
        "car free pedestrian street",
        "building entrance directly connected to sidewalk",
        "accessible pedestrian route",
        "bike racks near a building",
        "bicycle parking facilities",
        "covered bicycle parking",
        "secure bike storage room",
        "bike lane next to a building",
        "cyclist using a bike lane near a building",
        "bicycle repair station",
        "reduced parking footprint",
        "limited car parking with large pedestrian areas",
        "shared parking area",
        "underground parking reducing surface parking",
        "electric vehicle charging station",
        "EV charging point in parking area",
        "green vehicle parking space",
        "preferred parking for low emission vehicles",
        "dense urban neighborhood",
        "mixed use street with shops and housing",
        "building surrounded by diverse uses",
        "shops and services within walking distance",
        "active street frontage with pedestrians",
        "compact urban development",
        "connected street network",
        "walkable campus with connected paths",
        "site supporting walking biking and transit"
    ],

    "Not Clearly LEED Related": [
        "random object with no building feature",
        "plain wall with no sustainability feature",
        "ordinary furniture with no environmental strategy",
        "decorative image",
        "food on a table",
        "person standing with no building feature",
        "landscape with no building or transportation feature",
        "empty room with no windows or visible systems",
        "normal building with no visible sustainable strategy",
        "abstract image unrelated to buildings",
        "ordinary street with no pedestrian transit or bicycle feature",
        "interior space with no daylight ventilation or comfort feature",
        "parking lot with no green vehicle or pedestrian strategy",
        "building photo with no visible LEED related feature"
    ]
}


def get_explanation(category):
    if category == "Energy and Atmosphere":
        return (
            "The image appears to relate to Energy and Atmosphere because it may show "
            "energy reduction, renewable energy, efficient lighting, passive shading, "
            "building envelope performance, HVAC efficiency, energy metering, or controls. "
            "This is visual evidence only, not official LEED compliance."
        )

    elif category == "Indoor Environmental Quality":
        return (
            "The image appears to relate to Indoor Environmental Quality because it may show "
            "daylight, views, natural ventilation, fresh air supply, thermal comfort, lighting comfort, "
            "acoustic treatment, low-emitting materials, or occupant control. "
            "This is visual evidence only, not official LEED compliance."
        )

    elif category == "Location and Transportation":
        return (
            "The image appears to relate to Location and Transportation because it may show "
            "public transit access, walkability, bicycle facilities, reduced parking, electric vehicle "
            "charging, compact urban context, or connection to surrounding services. "
            "This is visual evidence only, not official LEED compliance."
        )

    elif category == "Mixed LEED Evidence":
        return (
            "The image appears to contain more than one type of LEED-related evidence."
        )

    else:
        return (
            "The image does not show enough clear visual evidence to connect it confidently "
            "to the selected LEED categories."
        )


def classify_image(image):
    image = image.convert("RGB")

    prompts = []
    prompt_category = {}

    for category, descriptions in leed_categories.items():
        for description in descriptions:
            prompt = f"a photo showing {description}"
            prompts.append(prompt)
            prompt_category[prompt] = category

    inputs = processor(
        text=prompts,
        images=image,
        return_tensors="pt",
        padding=True
    )

    with torch.no_grad():
        outputs = model(**inputs)
        probabilities = outputs.logits_per_image.softmax(dim=1)[0]

    category_scores = {}

    for prompt, probability in zip(prompts, probabilities):
        category = prompt_category[prompt]
        category_scores[category] = category_scores.get(category, 0) + probability.item()

    total = sum(category_scores.values())

    for category in category_scores:
        category_scores[category] = category_scores[category] / total * 5

    sorted_scores = sorted(category_scores.items(), key=lambda x: x[1], reverse=True)

    best_category = sorted_scores[0][0]
    best_score = sorted_scores[0][1]
    second_score = sorted_scores[1][1]

    if best_category == "Not Clearly LEED Related" or best_score < 2.5:
        final_result = "Not Clearly LEED Related"
    elif abs(best_score - second_score) < 0.5:
        final_result = "Mixed LEED Evidence"
    else:
        final_result = best_category

    return final_result, best_score, sorted_scores


st.markdown('<div class="title">LEED Visual Evidence Classifier</div>', unsafe_allow_html=True)

st.markdown(
    '<div class="subtitle">Upload a building-related image and classify it under selected LEED BD+C categories.</div>',
    unsafe_allow_html=True
)

left_col, right_col = st.columns([1, 1])

with left_col:
    st.subheader("Upload Image")

    uploaded_file = st.file_uploader(
        "Choose an image",
        type=["jpg", "jpeg", "png"]
    )

    st.markdown(
        """
        **Categories checked:**

        - Energy and Atmosphere  
        - Indoor Environmental Quality  
        - Location and Transportation  
        - Not Clearly LEED Related
        """
    )

    st.info(
        "This tool gives a visual evidence rating. It does not prove official LEED compliance."
    )

with right_col:
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_container_width=True)

        if st.button("Analyze Image"):
            final_result, best_score, sorted_scores = classify_image(image)

            st.markdown('<div class="result-box">', unsafe_allow_html=True)

            st.markdown("### Final Result")
            st.markdown(f'<div class="category">{final_result}</div>', unsafe_allow_html=True)

            st.write(get_explanation(final_result))

            st.markdown("---")

            st.markdown("### Rating Scores")

            for category, score in sorted_scores:
                st.write(f"**{category}:** {score:.2f} / 5")
                st.progress(min(score / 5, 1.0))

            st.markdown(
                '<div class="note">Scores are based on visible image evidence only.</div>',
                unsafe_allow_html=True
            )

            st.markdown('</div>', unsafe_allow_html=True)

    else:
        st.warning("Upload an image first to start the analysis.")