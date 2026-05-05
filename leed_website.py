import streamlit as st
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
import torch
import pandas as pd
from datetime import datetime
from pathlib import Path
import base64


# -----------------------------
# Local Media Helper
# -----------------------------
def get_base64_media(file_path):
    """
    Converts a local media file into base64 so it can be used as a CSS background.
    The GIF file must be in the same folder as leed_website.py.
    """
    path = Path(file_path)
    if path.exists():
        return base64.b64encode(path.read_bytes()).decode()
    return None


# -----------------------------
# Page Configuration
# -----------------------------
st.set_page_config(
    page_title="LEED Visual Evidence Classifier",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded"
)


# -----------------------------
# Custom CSS — Swiss / International Typographic Style
# -----------------------------
st.markdown(
    """
    <style>
        :root {
            --bg: #f2f1ec;
            --paper: #fdfcf8;
            --ink: #111111;
            --muted: #555555;
            --line: #111111;
            --soft-line: #d8d6cd;
            --green: #2f5f3a;
            --red: #b13a2f;
            --amber: #a26b00;
            --blue: #1f4d7a;
        }

        html, body, [class*="css"] {
            font-family: Helvetica, Arial, sans-serif;
            color: var(--ink) !important;
        }

        .stApp {
            background: var(--bg) !important;
            color: var(--ink) !important;
        }

        section[data-testid="stSidebar"] {
            background: #e7e5dd !important;
            border-right: 1.5px solid var(--line);
        }

        section[data-testid="stSidebar"] * {
            color: var(--ink) !important;
        }

        h1, h2, h3, h4, h5, h6, p, span, div, label {
            color: var(--ink) !important;
        }

        .hero {
            display: grid;
            grid-template-columns: 1fr;
            gap: 18px;
            padding: 34px 0 28px 0;
            border-bottom: 2px solid var(--line);
            margin-bottom: 28px;
            background: transparent;
        }

        .eyebrow {
            display: inline-block;
            width: fit-content;
            padding: 0;
            background: transparent;
            color: var(--green) !important;
            font-size: 12px;
            font-weight: 800;
            letter-spacing: 0.11em;
            text-transform: uppercase;
        }

        .hero-title {
            max-width: 980px;
            font-size: clamp(46px, 7vw, 94px);
            line-height: 0.92;
            font-weight: 900;
            letter-spacing: -0.065em;
            color: var(--ink) !important;
            margin: 0;
        }

        .hero-subtitle {
            max-width: 780px;
            font-size: 17px;
            line-height: 1.55;
            color: var(--muted) !important;
            border-left: 6px solid var(--green);
            padding-left: 18px;
        }

        .glass-card {
            padding: 24px;
            border-radius: 0;
            background: rgba(253, 252, 248, 0.94) !important;
            border: 1.5px solid var(--line);
            box-shadow: none;
            height: 100%;
            backdrop-filter: blur(2px);
        }

        .result-card {
            padding: 26px;
            border-radius: 0;
            background: rgba(253, 252, 248, 0.95) !important;
            border: 1.5px solid var(--line);
            box-shadow: none;
            margin-top: 24px;
            backdrop-filter: blur(2px);
        }

        .result-label {
            color: var(--green) !important;
            font-size: 12px;
            font-weight: 900;
            letter-spacing: 0.12em;
            text-transform: uppercase;
            margin-bottom: 9px;
        }

        .result-title {
            font-size: clamp(32px, 4vw, 56px);
            font-weight: 900;
            letter-spacing: -0.055em;
            color: var(--ink) !important;
            margin-bottom: 10px;
            line-height: 0.98;
        }

        .section-title {
            font-size: 24px;
            font-weight: 900;
            letter-spacing: -0.04em;
            color: var(--ink) !important;
            margin: 0 0 14px 0;
            border-bottom: 1.5px solid var(--line);
            padding-bottom: 10px;
        }

        .small-muted {
            color: var(--muted) !important;
            font-size: 14px;
            line-height: 1.55;
        }

        .pill {
            display: inline-block;
            padding: 7px 10px;
            border-radius: 0;
            background: transparent !important;
            color: var(--green) !important;
            border: 1.5px solid var(--green);
            font-size: 12px;
            font-weight: 800;
            margin: 4px 6px 4px 0;
            text-transform: uppercase;
            letter-spacing: 0.04em;
        }

        .warn-pill {
            display: inline-block;
            padding: 7px 10px;
            border-radius: 0;
            background: transparent !important;
            color: var(--amber) !important;
            border: 1.5px solid var(--amber);
            font-size: 12px;
            font-weight: 800;
            margin: 4px 6px 4px 0;
            text-transform: uppercase;
            letter-spacing: 0.04em;
        }

        .bad-pill {
            display: inline-block;
            padding: 7px 10px;
            border-radius: 0;
            background: transparent !important;
            color: var(--red) !important;
            border: 1.5px solid var(--red);
            font-size: 12px;
            font-weight: 800;
            margin: 4px 6px 4px 0;
            text-transform: uppercase;
            letter-spacing: 0.04em;
        }

        .divider {
            height: 1.5px;
            background: var(--line);
            margin: 22px 0;
        }

        div[data-testid="stMetric"] {
            background: rgba(253, 252, 248, 0.95) !important;
            border: 1.5px solid var(--line);
            border-radius: 0;
            padding: 16px;
            box-shadow: none;
        }

        div[data-testid="stMetric"] * {
            color: var(--ink) !important;
        }

        .stButton > button {
            border-radius: 0;
            padding: 0.82rem 1.45rem;
            border: 1.5px solid var(--line);
            background: var(--ink) !important;
            color: #ffffff !important;
            font-weight: 900;
            letter-spacing: 0.03em;
            text-transform: uppercase;
            box-shadow: none;
        }

        .stButton > button:hover {
            background: var(--green) !important;
            color: #ffffff !important;
            border: 1.5px solid var(--green);
        }

        div[data-testid="stAlert"] {
            border-radius: 0;
            border: 1.5px solid var(--line);
            background: #fffdf2 !important;
            color: var(--ink) !important;
        }

        div[data-testid="stFileUploader"] {
            background: rgba(255, 255, 255, 0.94) !important;
            border: 1.5px dashed var(--line);
            padding: 12px;
        }

        div[data-testid="stFileUploader"] * {
            color: var(--ink) !important;
        }

        .stTabs [data-baseweb="tab-list"] {
            gap: 0;
            border-bottom: 1.5px solid var(--line);
        }

        .stTabs [data-baseweb="tab"] {
            border-radius: 0;
            color: var(--ink) !important;
            font-weight: 800;
            border: 1.5px solid var(--line);
            border-bottom: 0;
            background: #ebe9df;
            padding: 10px 14px;
        }

        .stTabs [aria-selected="true"] {
            background: var(--paper) !important;
        }

        .stProgress > div > div > div > div {
            background-color: var(--green) !important;
        }

        div[data-testid="stDataFrame"] {
            border: 1.5px solid var(--line);
        }

        textarea, input {
            color: var(--ink) !important;
            background: #ffffff !important;
        }
    </style>
    """,
    unsafe_allow_html=True
)


# -----------------------------
# GIF Background
# -----------------------------
bg_gif = get_base64_media("leed_motion.gif")

if bg_gif:
    st.markdown(
        f"""
        <style>
            .stApp {{
                background:
                    linear-gradient(rgba(242, 241, 236, 0.45), rgba(242, 241, 236, 0.45)),
                    url("data:image/gif;base64,{bg_gif}") center center / cover fixed no-repeat !important;
                color: var(--ink) !important;
            }}
        </style>
        """,
        unsafe_allow_html=True
    )
else:
    st.warning("Background GIF not found. Make sure leed_motion.gif is in the same folder as leed_website.py.")


# -----------------------------
# Model Loading
# -----------------------------
@st.cache_resource(show_spinner="Loading the image-recognition model...")
def load_model():
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    return model, processor


model, processor = load_model()


# -----------------------------
# Criteria System
# -----------------------------
CRITERIA = {
    "Energy and Atmosphere": {
        "Renewable Energy": {
            "weight": 1.25,
            "positive": [
                "solar panels clearly visible on a building roof",
                "photovoltaic panels mounted on a facade",
                "solar canopy over an outdoor area",
                "solar water heating panels on a roof",
                "renewable energy equipment attached to a building"
            ],
            "negative": []
        },
        "Solar Control and Shading": {
            "weight": 1.10,
            "positive": [
                "deep overhangs shading windows",
                "external louvers shading a building facade",
                "brise soleil on a sun exposed facade",
                "perforated screen shading windows",
                "recessed windows protected by thick walls",
                "shaded courtyard reducing solar heat gain"
            ],
            "negative": [
                "unshaded glass facade in harsh direct sunlight",
                "large exposed windows with no shading devices"
            ]
        },
        "High Performance Envelope": {
            "weight": 1.00,
            "positive": [
                "double skin facade on a building",
                "insulated wall assembly visible in construction",
                "green roof on a building",
                "white reflective roof surface",
                "thick thermal mass wall with small openings",
                "high performance glazing with exterior shading"
            ],
            "negative": []
        },
        "Efficient Lighting and Controls": {
            "weight": 0.85,
            "positive": [
                "LED lighting fixtures clearly visible indoors",
                "occupancy sensor controlling lights",
                "daylight sensor for lighting control",
                "smart lighting control panel",
                "task lighting used instead of excessive general lighting"
            ],
            "negative": [
                "overly bright artificial lighting causing glare"
            ]
        },
        "Visible Mechanical Efficiency": {
            "weight": 0.90,
            "positive": [
                "visible high efficiency HVAC equipment",
                "visible air handling unit in a mechanical room",
                "visible chiller or heat pump equipment",
                "visible heat recovery ventilation unit",
                "exposed ductwork and ventilation equipment",
                "rooftop mechanical units clearly visible"
            ],
            "negative": [
                "plain building facade with no visible mechanical equipment",
                "interior room with no visible HVAC system"
            ]
        },
        "Energy Metering and Controls": {
            "weight": 0.95,
            "positive": [
                "building energy meter clearly visible",
                "energy monitoring screen in a building",
                "building management system display",
                "smart thermostat on a wall",
                "electricity control panel for building systems"
            ],
            "negative": []
        }
    },

    "Indoor Environmental Quality": {
        "Daylight Quality": {
            "weight": 1.05,
            "positive": [
                "interior space with soft natural daylight",
                "large windows providing balanced daylight",
                "skylight bringing controlled daylight indoors",
                "clerestory windows providing daylight",
                "light shelves reflecting daylight deeper into a room",
                "daylit interior with blinds controlling sunlight"
            ],
            "negative": [
                "harsh direct sunlight causing glare inside a room",
                "overexposed bright window glare",
                "strong sun patches on desks causing visual discomfort",
                "uncontrolled daylight glare in an interior space"
            ]
        },
        "Quality Views": {
            "weight": 0.90,
            "positive": [
                "interior workspace with clear view to outside",
                "large windows with outdoor landscape view",
                "classroom with view to trees or open space",
                "seating area visually connected to nature",
                "interior space overlooking a courtyard"
            ],
            "negative": [
                "window facing a blank wall with no quality view"
            ]
        },
        "Natural Ventilation and Fresh Air": {
            "weight": 1.15,
            "positive": [
                "operable windows for natural ventilation",
                "open windows allowing fresh air into a room",
                "cross ventilation through opposite openings",
                "small ventilation openings in a thick wall",
                "ceiling fans improving air movement",
                "fresh air supply diffusers visible in ceiling"
            ],
            "negative": [
                "sealed interior room with no visible ventilation",
                "closed windows with no visible fresh air strategy"
            ]
        },
        "Thermal Comfort and User Control": {
            "weight": 0.95,
            "positive": [
                "thermostat for occupant temperature control",
                "individual thermal comfort control",
                "ceiling fan for thermal comfort",
                "interior blinds controlling heat and glare",
                "comfortable indoor workspace with environmental controls"
            ],
            "negative": [
                "unshaded hot interior space",
                "direct sun hitting occupants without shading"
            ]
        },
        "Indoor Air Quality Materials": {
            "weight": 0.80,
            "positive": [
                "indoor plants in a clean interior space",
                "green wall inside a building",
                "air quality monitor in a room",
                "CO2 sensor in an interior space",
                "entry mat system reducing dirt entering a building",
                "clean interior with low emitting materials"
            ],
            "negative": [
                "dusty interior construction space",
                "visible smoke inside a room"
            ]
        },
        "Lighting Comfort": {
            "weight": 0.85,
            "positive": [
                "adjustable task lighting at workstations",
                "dimmable lighting controls",
                "soft indirect interior lighting",
                "well lit workspace without glare",
                "individual lighting controls"
            ],
            "negative": [
                "harsh overhead lighting causing glare",
                "dark interior workspace with poor lighting"
            ]
        },
        "Acoustic Comfort": {
            "weight": 0.90,
            "positive": [
                "acoustic ceiling panels visible in a room",
                "sound absorbing wall panels",
                "perforated acoustic ceiling",
                "wood acoustic panels in an auditorium",
                "hanging acoustic baffles",
                "soft surfaces reducing noise in a room"
            ],
            "negative": [
                "large hard empty room with no acoustic treatment"
            ]
        }
    },

    "Location and Transportation": {
        "Public Transit Access": {
            "weight": 1.20,
            "positive": [
                "bus stop near a building entrance",
                "metro station entrance near a building",
                "tram stop near a building",
                "train station near a building",
                "transit shelter on a walkable street",
                "people walking from transit to a building"
            ],
            "negative": [
                "isolated building with no visible transit access"
            ]
        },
        "Walkability and Pedestrian Access": {
            "weight": 1.05,
            "positive": [
                "wide sidewalk connected to a building entrance",
                "safe pedestrian crossing leading to a building",
                "shaded pedestrian path",
                "pedestrian plaza in front of a building",
                "car free pedestrian street",
                "accessible pedestrian route to a building"
            ],
            "negative": [
                "building surrounded by roads with no sidewalk",
                "unsafe pedestrian environment dominated by cars"
            ]
        },
        "Bicycle Facilities": {
            "weight": 1.05,
            "positive": [
                "bike racks near a building",
                "covered bicycle parking",
                "secure bicycle storage room",
                "bike lane next to a building",
                "bicycle repair station",
                "cyclist using a bike lane near a building"
            ],
            "negative": []
        },
        "Reduced Parking and Car Dependence": {
            "weight": 0.90,
            "positive": [
                "small parking area with large pedestrian space",
                "reduced surface parking footprint",
                "underground parking reducing surface parking",
                "landscape replacing surface parking",
                "pedestrian first site plan with limited cars"
            ],
            "negative": [
                "large surface parking lot dominating the site",
                "building surrounded by expansive car parking"
            ]
        },
        "Green Vehicles": {
            "weight": 0.95,
            "positive": [
                "electric vehicle charging station",
                "EV charging point in a parking area",
                "preferred parking for low emission vehicles",
                "electric car charging near a building"
            ],
            "negative": []
        },
        "Compact Mixed Use Context": {
            "weight": 0.85,
            "positive": [
                "dense urban neighborhood around a building",
                "mixed use street with shops and housing",
                "active street frontage with pedestrians",
                "shops and services within walking distance",
                "compact urban development"
            ],
            "negative": [
                "isolated building in an undeveloped area"
            ]
        }
    }
}


UNRELATED_PROMPTS = [
    "random object with no building feature",
    "food on a table",
    "person standing with no building feature",
    "abstract image unrelated to buildings",
    "plain wall with no sustainability feature",
    "ordinary furniture with no environmental strategy",
    "normal building photo with no visible sustainability strategy"
]


CATEGORY_ICONS = {
    "Energy and Atmosphere": "⚡",
    "Indoor Environmental Quality": "🌤️",
    "Location and Transportation": "🚲",
    "Mixed LEED Evidence": "🔀",
    "Not Clearly LEED Related": "—"
}


# -----------------------------
# Scoring Helpers
# -----------------------------
def clip_similarity(image, texts):
    """
    Return CLIP cosine similarities for image against a list of text prompts.
    This Streamlit Cloud-safe version uses model(**inputs) and reads
    outputs.image_embeds / outputs.text_embeds directly.
    """

    inputs = processor(
        text=texts,
        images=image,
        return_tensors="pt",
        padding=True,
        truncation=True
    )

    with torch.no_grad():
        outputs = model(**inputs)
        image_features = outputs.image_embeds
        text_features = outputs.text_embeds

    image_features = torch.nn.functional.normalize(image_features, p=2, dim=-1)
    text_features = torch.nn.functional.normalize(text_features, p=2, dim=-1)

    similarities = torch.matmul(image_features, text_features.T)[0]

    return similarities.detach().cpu().tolist()


def normalize_score(similarity, unrelated_baseline):
    """
    Convert CLIP similarity into a 0-5 visual evidence score.
    This is intentionally conservative to reduce false positives.
    """
    adjusted = similarity - unrelated_baseline
    score = (adjusted - 0.015) / 0.095 * 5
    return max(0, min(5, score))


def analyze_image(image):
    image = image.convert("RGB")

    unrelated_scores = clip_similarity(
        image,
        [f"a photo of {p}" for p in UNRELATED_PROMPTS]
    )

    unrelated_baseline = max(unrelated_scores)

    category_results = {}
    all_detected = []
    all_warnings = []

    for category, criteria in CRITERIA.items():
        criterion_rows = []
        weighted_total = 0
        weight_sum = 0

        for criterion_name, data in criteria.items():
            positive_prompts = [f"a photo showing {p}" for p in data["positive"]]
            positive_scores = clip_similarity(image, positive_prompts)

            best_pos_score = max(positive_scores)
            best_pos_prompt = data["positive"][positive_scores.index(best_pos_score)]

            positive_rating = normalize_score(best_pos_score, unrelated_baseline)

            negative_rating = 0
            best_neg_prompt = None

            if data.get("negative"):
                negative_prompts = [f"a photo showing {p}" for p in data["negative"]]
                negative_scores = clip_similarity(image, negative_prompts)

                best_neg_score = max(negative_scores)
                best_neg_prompt = data["negative"][negative_scores.index(best_neg_score)]

                negative_rating = normalize_score(best_neg_score, unrelated_baseline)

            final_rating = max(0, positive_rating - (0.45 * negative_rating))

            if final_rating < 1.15:
                final_rating = 0

            weighted_total += final_rating * data["weight"]
            weight_sum += data["weight"]

            status = "Not detected"

            if final_rating >= 3.5:
                status = "Strong visual evidence"
            elif final_rating >= 2.3:
                status = "Moderate visual evidence"
            elif final_rating >= 1.15:
                status = "Weak visual evidence"

            warning = None

            if negative_rating >= 2.4:
                warning = best_neg_prompt
                all_warnings.append({
                    "Category": category,
                    "Issue": criterion_name,
                    "Visual caution": best_neg_prompt,
                    "Caution score": round(negative_rating, 2)
                })

            row = {
                "Category": category,
                "Criterion": criterion_name,
                "Score": round(final_rating, 2),
                "Status": status,
                "Best visual match": best_pos_prompt,
                "Visual caution": warning or "—"
            }

            criterion_rows.append(row)

            if final_rating >= 2.3:
                all_detected.append(row)

        category_score = weighted_total / weight_sum if weight_sum else 0

        category_results[category] = {
            "score": round(category_score, 2),
            "criteria": criterion_rows
        }

    sorted_categories = sorted(
        [(cat, data["score"]) for cat, data in category_results.items()],
        key=lambda x: x[1],
        reverse=True
    )

    top_category, top_score = sorted_categories[0]
    second_category, second_score = sorted_categories[1]

    if top_score < 2.0:
        final_result = "Not Clearly LEED Related"
    elif abs(top_score - second_score) <= 0.45 and second_score >= 1.8:
        final_result = "Mixed LEED Evidence"
    else:
        final_result = top_category

    return {
        "final_result": final_result,
        "top_category": top_category,
        "top_score": top_score,
        "second_category": second_category,
        "second_score": second_score,
        "category_results": category_results,
        "sorted_categories": sorted_categories,
        "detected": all_detected,
        "warnings": all_warnings,
        "unrelated_baseline": round(unrelated_baseline, 3)
    }


def explanation_for_result(result, analysis):
    if result == "Energy and Atmosphere":
        return (
            "The image has visible evidence linked to energy reduction, passive solar control, "
            "renewable energy, efficient lighting, envelope performance, metering, or clearly visible "
            "mechanical systems."
        )

    if result == "Indoor Environmental Quality":
        return (
            "The image has visible evidence linked to occupant comfort, daylight quality, views, "
            "ventilation, air quality, thermal comfort, lighting comfort, or acoustics. Window evidence "
            "is treated carefully because uncontrolled glare can reduce the IEQ score."
        )

    if result == "Location and Transportation":
        return (
            "The image has visible evidence linked to walkability, public transit access, bicycle facilities, "
            "reduced parking, EV charging, or compact mixed-use context."
        )

    if result == "Mixed LEED Evidence":
        return (
            f"The image appears to support more than one LEED category. The closest categories are "
            f"{analysis['top_category']} and {analysis['second_category']}."
        )

    return (
        "The image does not show enough clear visible evidence to confidently connect it to the selected "
        "LEED categories."
    )


def build_report_text(analysis):
    lines = []

    lines.append("LEED Visual Evidence Classifier Report")
    lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    lines.append("")
    lines.append(f"Final result: {analysis['final_result']}")
    lines.append(f"Top category: {analysis['top_category']} ({analysis['top_score']}/5)")
    lines.append(f"Second category: {analysis['second_category']} ({analysis['second_score']}/5)")
    lines.append("")

    lines.append("Category scores:")

    for cat, score in analysis["sorted_categories"]:
        lines.append(f"- {cat}: {score}/5")

    lines.append("")
    lines.append("Detected visual evidence:")

    if analysis["detected"]:
        for item in analysis["detected"]:
            lines.append(
                f"- {item['Category']} / {item['Criterion']}: "
                f"{item['Score']}/5 — {item['Best visual match']}"
            )
    else:
        lines.append("- No strong visual evidence detected.")

    lines.append("")
    lines.append("Cautions:")

    if analysis["warnings"]:
        for item in analysis["warnings"]:
            lines.append(
                f"- {item['Category']} / {item['Issue']}: "
                f"{item['Visual caution']}"
            )
    else:
        lines.append("- No major visual cautions detected.")

    lines.append("")
    lines.append(
        "Limitations: This is a visual evidence tool only. It does not prove official LEED compliance, "
        "which requires documentation, calculations, drawings, product data, and verification."
    )

    return "\n".join(lines)


# -----------------------------
# Sidebar
# -----------------------------
with st.sidebar:
    st.markdown("### 🌿 LEED Visual Tool")
    st.write(
        "Upload a building-related image and evaluate it as visual evidence "
        "for selected LEED BD+C categories."
    )

    st.markdown("---")

    st.markdown("**Categories**")
    st.markdown("- ⚡ Energy and Atmosphere")
    st.markdown("- 🌤️ Indoor Environmental Quality")
    st.markdown("- 🚲 Location and Transportation")

    st.markdown("---")

    st.caption(
        "This app is intentionally conservative. It avoids giving full credit for vague images, "
        "generic windows, or invisible systems."
    )


# -----------------------------
# Header
# -----------------------------
st.markdown(
    """
    <div class="hero">
        <div class="eyebrow">AI image analysis · LEED visual evidence</div>
        <div class="hero-title">LEED Visual Evidence Classifier</div>
        <div class="hero-subtitle">
            An image-based tool that rates visible sustainability strategies under selected
            LEED BD+C categories. It does not certify LEED compliance — it organizes visual evidence
            and flags uncertainty in image, and weak evidence.
        </div>
    </div>
    """,
    unsafe_allow_html=True
)


# -----------------------------
# Main Layout
# -----------------------------
left_col, right_col = st.columns([0.95, 1.25], gap="large")


with left_col:
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)

    st.markdown('<div class="section-title">Upload Image</div>', unsafe_allow_html=True)

    st.markdown(
        '<div class="small-muted">'
        'Use a clear photo of a building, interior, facade, street edge, site, system, or detail.'
        '</div>',
        unsafe_allow_html=True
    )

    uploaded_file = st.file_uploader(
        "Choose a JPG or PNG image",
        type=["jpg", "jpeg", "png"]
    )

    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

    st.markdown('<div class="section-title">Improved Logic</div>', unsafe_allow_html=True)

    st.markdown('<span class="pill">Sub-criteria scoring</span>', unsafe_allow_html=True)
    st.markdown('<span class="pill">Glare penalty</span>', unsafe_allow_html=True)
    st.markdown('<span class="pill">Conservative HVAC detection</span>', unsafe_allow_html=True)
    st.markdown('<span class="pill">Visible evidence only</span>', unsafe_allow_html=True)

    st.info(
        "Tip: A normal window is not automatically a strong IEQ score. "
        "The app checks for daylight quality and also looks for glare risk."
    )

    st.markdown('</div>', unsafe_allow_html=True)


with right_col:
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)

    if uploaded_file is None:
        st.markdown('<div class="section-title">Preview</div>', unsafe_allow_html=True)
        st.warning("Upload an image to start the analysis.")

    else:
        image = Image.open(uploaded_file)

        st.image(
            image,
            caption="Uploaded image",
            width="stretch"
        )

        analyze_button = st.button(
            "Analyze Image",
            use_container_width=True
        )

        if analyze_button:
            with st.spinner("Analyzing visible LEED evidence..."):
                analysis = analyze_image(image)
                st.session_state["analysis"] = analysis

    st.markdown('</div>', unsafe_allow_html=True)


# -----------------------------
# Results
# -----------------------------
if "analysis" in st.session_state:
    analysis = st.session_state["analysis"]
    final_result = analysis["final_result"]

    st.markdown('<div class="result-card">', unsafe_allow_html=True)

    st.markdown('<div class="result-label">Final result</div>', unsafe_allow_html=True)

    st.markdown(
        f'<div class="result-title">{CATEGORY_ICONS.get(final_result, "")} {final_result}</div>',
        unsafe_allow_html=True
    )

    st.write(explanation_for_result(final_result, analysis))

    score_cols = st.columns(3)

    for i, (cat, score) in enumerate(analysis["sorted_categories"]):
        with score_cols[i]:
            st.metric(
                label=f"{CATEGORY_ICONS.get(cat, '')} {cat}",
                value=f"{score}/5"
            )

    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

    tabs = st.tabs([
        "Score Breakdown",
        "Detected Evidence",
        "Cautions",
        "Limitations",
        "Download Report"
    ])

    with tabs[0]:
        st.markdown("### Category score bars")

        for cat, score in analysis["sorted_categories"]:
            st.write(f"**{CATEGORY_ICONS.get(cat, '')} {cat}: {score}/5**")
            st.progress(min(score / 5, 1.0))

        st.markdown("### Detailed criteria table")

        rows = []

        for cat, data in analysis["category_results"].items():
            rows.extend(data["criteria"])

        df = pd.DataFrame(rows)

        st.dataframe(
            df,
            use_container_width=True,
            hide_index=True
        )

    with tabs[1]:
        st.markdown("### Visible evidence detected")

        if analysis["detected"]:
            for item in sorted(analysis["detected"], key=lambda x: x["Score"], reverse=True):
                st.markdown(
                    f'<span class="pill">{item["Category"]} · {item["Criterion"]} · {item["Score"]}/5</span>',
                    unsafe_allow_html=True
                )

                st.write(f"Best visual match: {item['Best visual match']}")
        else:
            st.write("No moderate or strong visual evidence was detected.")

    with tabs[2]:
        st.markdown("### Visual cautions")

        if analysis["warnings"]:
            for item in analysis["warnings"]:
                st.markdown(
                    f'<span class="warn-pill">{item["Category"]} · {item["Issue"]}</span>',
                    unsafe_allow_html=True
                )

                st.write(f"Caution: {item['Visual caution']}")
        else:
            st.success("No major visual cautions detected.")

    with tabs[3]:
        st.markdown("### What this tool can and cannot do")

        st.markdown(
            """
            **This tool can:**

            - Sort a photo under likely LEED-related visual categories.
            - Identify visible evidence such as solar panels, shading devices, bike racks, daylight, acoustic panels, or EV charging.
            - Flag risks such as glare or car-dominated site conditions.

            **This tool cannot:**

            - Prove official LEED compliance.
            - Confirm energy performance, HVAC efficiency, refrigerant management, daylight calculations, acoustic performance, or transit service frequency from an image alone.
            - Replace drawings, specifications, calculations, product data, or LEED documentation.
            """
        )

    with tabs[4]:
        report_text = build_report_text(analysis)

        st.text_area(
            "Report preview",
            report_text,
            height=300
        )

        st.download_button(
            label="Download report as TXT",
            data=report_text,
            file_name="leed_visual_evidence_report.txt",
            mime="text/plain",
            use_container_width=True
        )

    st.markdown('</div>', unsafe_allow_html=True)
