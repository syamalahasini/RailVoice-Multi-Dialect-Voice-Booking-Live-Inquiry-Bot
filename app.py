import streamlit as st
import pandas as pd
import whisper
import tempfile
import os
import re
from gtts import gTTS


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="RailVoice",
    page_icon="🚆",
    layout="wide"
)


# ============================================================
# TITLE
# ============================================================

st.title("🚆 RailVoice")
st.subheader("Multi-Dialect Voice Booking & Live Inquiry Bot")

st.write(
    "You can ask about trains, routes, train numbers, "
    "running days and timings."
)


# ============================================================
# LOAD WHISPER
# ============================================================

@st.cache_resource
def load_whisper_model():
    return whisper.load_model("base")


model = load_whisper_model()


# ============================================================
# LOAD TRAINS CSV
# ============================================================

@st.cache_data
def load_trains():

    file_path = "trains.csv"

    if not os.path.exists(file_path):
        return None

    df = pd.read_csv(file_path)

    df = df.dropna(how="all")

    # Clean column names
    df.columns = (
        df.columns
        .astype(str)
        .str.strip()
        .str.lower()
        .str.replace(" ", "_")
        .str.replace("-", "_")
    )

    return df


trains = load_trains()


# ============================================================
# CHECK CSV
# ============================================================

if trains is None:

    st.error(
        "❌ trains.csv was not found.\n\n"
        "Make sure trains.csv is in the same folder as app.py."
    )

    st.stop()


# ============================================================
# FIND COLUMN
# ============================================================

def get_column(columns, possible_names):

    for name in possible_names:

        if name in columns:
            return name

    return None


# ============================================================
# DETECT CSV COLUMNS
# ============================================================

TRAIN_NO_COL = get_column(
    trains.columns,
    [
        "train_no",
        "train_number",
        "train_num",
        "number",
        "train"
    ]
)

TRAIN_NAME_COL = get_column(
    trains.columns,
    [
        "train_name",
        "name"
    ]
)

SOURCE_COL = get_column(
    trains.columns,
    [
        "source",
        "from",
        "origin",
        "source_station",
        "start_station",
        "departure_station"
    ]
)

DESTINATION_COL = get_column(
    trains.columns,
    [
        "destination",
        "dest",
        "to",
        "destination_station",
        "end_station",
        "arrival_station"
    ]
)

DEPARTURE_COL = get_column(
    trains.columns,
    [
        "departure",
        "departure_time",
        "depart",
        "start_time",
        "dep_time"
    ]
)

ARRIVAL_COL = get_column(
    trains.columns,
    [
        "arrival",
        "arrival_time",
        "arrive",
        "end_time",
        "arr_time"
    ]
)

RUNS_COL = get_column(
    trains.columns,
    [
        "runs_days",
        "running_days",
        "days",
        "run_days",
        "runs_on"
    ]
)


# ============================================================
# STATION ALIASES
# ============================================================

STATION_ALIASES = {

    "delhi": "delhi",
    "new delhi": "new delhi",

    "mumbai": "mumbai",
    "bombay": "mumbai",

    "hyderabad": "hyderabad",

    "bangalore": "bengaluru",
    "bengaluru": "bengaluru",
    "bengalore": "bengaluru",

    "chennai": "chennai",
    "madras": "chennai",

    "kolkata": "kolkata",
    "calcutta": "kolkata",

    "visakhapatnam": "visakhapatnam",
    "vishakapad": "visakhapatnam",
    "vishakapatnam": "visakhapatnam",
    "vishakhapatnam": "visakhapatnam",
    "vizag": "visakhapatnam",

    "sikandrabad": "sikandrabad",
    "sikandrabha": "sikandrabad",

    "pune": "pune",
    "ahmedabad": "ahmedabad",
    "jaipur": "jaipur",
    "lucknow": "lucknow",
    "bhopal": "bhopal",
    "patna": "patna",
    "varanasi": "varanasi",
    "agra": "agra",
    "surat": "surat",
    "nagpur": "nagpur",
    "coimbatore": "coimbatore",
    "kochi": "kochi",
    "ernakulam": "ernakulam"
}


# ============================================================
# NORMALIZE STATION
# ============================================================

def normalize_station(name):

    name = str(name).lower().strip()

    name = re.sub(
        r"[^a-z0-9\s]",
        "",
        name
    )

    name = re.sub(
        r"\s+",
        " ",
        name
    )

    if name in STATION_ALIASES:
        return STATION_ALIASES[name]

    return name


# ============================================================
# NORMALIZE CSV STATION
# ============================================================

def normalize_csv_station(name):

    name = str(name).lower().strip()

    name = re.sub(
        r"[^a-z0-9\s]",
        "",
        name
    )

    name = re.sub(
        r"\s+",
        " ",
        name
    )

    return name


# ============================================================
# TRAIN NUMBER EXTRACTION
# ============================================================

def extract_train_number(text):

    text = text.lower().strip()

    # --------------------------------------------------------
    # CASE 1: Normal number
    # Example:
    # "Tell me about train 12806"
    # --------------------------------------------------------

    match = re.search(
        r"(?<!\d)\d{4,5}(?!\d)",
        text
    )

    if match:
        return match.group(0)


    # --------------------------------------------------------
    # CASE 2: Spaced digits
    # Example:
    # "Tell me about train 1 2 8 0 6"
    # --------------------------------------------------------

    spaced = re.search(
        r"\b(\d)\s+(\d)\s+(\d)\s+(\d)(?:\s+(\d))?\b",
        text
    )

    if spaced:

        digits = "".join(
            x for x in spaced.groups()
            if x is not None
        )

        if len(digits) in [4, 5]:
            return digits


    # --------------------------------------------------------
    # CASE 3: Number words
    # Example:
    # "train one two eight zero six"
    # --------------------------------------------------------

    number_words = {

        "zero": "0",
        "one": "1",
        "two": "2",
        "three": "3",
        "four": "4",
        "five": "5",
        "six": "6",
        "seven": "7",
        "eight": "8",
        "nine": "9"
    }

    words = text.split()

    digits = []

    for word in words:

        if word in number_words:

            digits.append(
                number_words[word]
            )

    if len(digits) in [4, 5]:

        return "".join(digits)

    return None


# ============================================================
# ROUTE EXTRACTION
# ============================================================

def extract_route(text):

    text = text.lower().strip()


    # --------------------------------------------------------
    # FROM X TO Y
    # --------------------------------------------------------

    pattern1 = (
        r"from\s+(.+?)\s+to\s+(.+?)(?:\.|$)"
    )

    match = re.search(
        pattern1,
        text
    )

    if match:

        source = match.group(1).strip()
        destination = match.group(2).strip()

        destination = re.sub(
            r"\b(train|trains|railway|railways|please)\b.*$",
            "",
            destination
        ).strip()

        return (
            normalize_station(source),
            normalize_station(destination)
        )


    # --------------------------------------------------------
    # X TO Y
    # --------------------------------------------------------

    pattern2 = (
        r"(.+?)\s+to\s+(.+?)(?:\.|$)"
    )

    match = re.search(
        pattern2,
        text
    )

    if match:

        source = match.group(1).strip()
        destination = match.group(2).strip()

        source = re.sub(
            r"^(show me|find|find me|tell me about|"
            r"are there|is there|any trains from)\s+",
            "",
            source
        )

        destination = re.sub(
            r"\b(train|trains|railway|railways|please)\b.*$",
            "",
            destination
        ).strip()

        return (
            normalize_station(source),
            normalize_station(destination)
        )


    # --------------------------------------------------------
    # BETWEEN X AND Y
    # --------------------------------------------------------

    pattern3 = (
        r"between\s+(.+?)\s+and\s+(.+?)(?:\.|$)"
    )

    match = re.search(
        pattern3,
        text
    )

    if match:

        source = normalize_station(
            match.group(1)
        )

        destination = normalize_station(
            match.group(2)
        )

        return source, destination


    return None


# ============================================================
# STATION MATCHING
# ============================================================

def station_matches(
    csv_value,
    requested
):

    csv_value = normalize_csv_station(
        csv_value
    )

    requested = normalize_station(
        requested
    )


    # Exact
    if csv_value == requested:
        return True


    # Partial
    if requested in csv_value:
        return True


    if csv_value in requested:
        return True


    # Delhi
    if requested == "delhi":

        if "delhi" in csv_value:
            return True


    # Mumbai
    if requested == "mumbai":

        if "mumbai" in csv_value:
            return True


    # Hyderabad
    if requested == "hyderabad":

        if "hyderabad" in csv_value:
            return True


    # Bengaluru
    if requested == "bengaluru":

        if (
            "bengaluru" in csv_value
            or "bangalore" in csv_value
        ):
            return True


    # Chennai
    if requested == "chennai":

        if (
            "chennai" in csv_value
            or "madras" in csv_value
        ):
            return True


    # Visakhapatnam
    if requested == "visakhapatnam":

        if (
            "visakhapatnam" in csv_value
            or "vishakhapatnam" in csv_value
            or "vishakapatnam" in csv_value
            or "vizag" in csv_value
        ):
            return True


    return False


# ============================================================
# SEARCH BY ROUTE
# ============================================================

def search_trains(
    source,
    destination
):

    if (
        SOURCE_COL is None
        or DESTINATION_COL is None
    ):

        return pd.DataFrame()


    matches = []


    for index, row in trains.iterrows():

        csv_source = row[SOURCE_COL]

        csv_destination = row[
            DESTINATION_COL
        ]


        if station_matches(
            csv_source,
            source
        ):

            if station_matches(
                csv_destination,
                destination
            ):

                matches.append(row)


    if not matches:

        return pd.DataFrame()


    return pd.DataFrame(
        matches
    )


# ============================================================
# SEARCH TRAIN NUMBER
# ============================================================

def search_train_number(
    train_number
):

    if TRAIN_NO_COL is None:

        return pd.DataFrame()


    requested = str(
        train_number
    ).strip()


    numbers = (
        trains[TRAIN_NO_COL]
        .fillna("")
        .astype(str)
        .str.replace(
            r"\.0$",
            "",
            regex=True
        )
        .str.replace(
            r"\D",
            "",
            regex=True
        )
    )


    result = trains[
        numbers == requested
    ]


    return result


# ============================================================
# FORMAT ROUTE RESULTS
# ============================================================

def format_results(
    result,
    source,
    destination
):

    if result.empty:

        return (
            f"I could not find any trains from "
            f"{source.title()} to "
            f"{destination.title()}."
        )


    response = (
        f"I found {len(result)} train"
        f"{'s' if len(result) != 1 else ''} "
        f"from {source.title()} to "
        f"{destination.title()}.\n\n"
    )


    # First 10 for audio
    voice_rows = result.head(10)


    for i, (_, row) in enumerate(
        voice_rows.iterrows(),
        start=1
    ):

        if TRAIN_NO_COL:

            train_number = str(
                row[TRAIN_NO_COL]
            )

        else:

            train_number = (
                "Not available"
            )


        if TRAIN_NAME_COL:

            train_name = str(
                row[TRAIN_NAME_COL]
            )

        else:

            train_name = (
                "Train name not available"
            )


        if DEPARTURE_COL:

            departure = str(
                row[DEPARTURE_COL]
            )

        else:

            departure = (
                "Not available"
            )


        if ARRIVAL_COL:

            arrival = str(
                row[ARRIVAL_COL]
            )

        else:

            arrival = (
                "Not available"
            )


        if RUNS_COL:

            runs = str(
                row[RUNS_COL]
            )

        else:

            runs = (
                "Not available"
            )


        response += (
            f"{i}. Train {train_number}, "
            f"{train_name}. "
            f"Departure {departure}. "
            f"Arrival {arrival}. "
            f"Runs on {runs}.\n"
        )


    if len(result) > 10:

        response += (
            f"There are {len(result)} trains "
            f"in total. I am listing the "
            f"first 10."
        )


    return response


# ============================================================
# PROCESS RAILWAY QUERY
# ============================================================

def process_railway_query(text):

    text_lower = text.lower().strip()


    # ========================================================
    # 1. TRAIN NUMBER
    # ========================================================

    train_number = extract_train_number(
        text
    )


    if train_number:

        result = search_train_number(
            train_number
        )


        if result.empty:

            return (
                f"I could not find train number "
                f"{train_number} in the train database."
            )


        row = result.iloc[0]


        train_name = (
            str(row[TRAIN_NAME_COL])
            if TRAIN_NAME_COL
            else "Unknown train"
        )


        source = (
            str(row[SOURCE_COL])
            if SOURCE_COL
            else "Unknown"
        )


        destination = (
            str(row[DESTINATION_COL])
            if DESTINATION_COL
            else "Unknown"
        )


        departure = (
            str(row[DEPARTURE_COL])
            if DEPARTURE_COL
            else "Not available"
        )


        arrival = (
            str(row[ARRIVAL_COL])
            if ARRIVAL_COL
            else "Not available"
        )


        runs = (
            str(row[RUNS_COL])
            if RUNS_COL
            else "Not available"
        )


        return (
            f"Train number {train_number} "
            f"is {train_name}. "
            f"It runs from {source} to "
            f"{destination}. "
            f"The departure time is "
            f"{departure}. "
            f"The arrival time is "
            f"{arrival}. "
            f"It runs on {runs}."
        )


    # ========================================================
    # 2. ROUTE
    # ========================================================

    route = extract_route(
        text
    )


    if route:

        source, destination = route


        result = search_trains(
            source,
            destination
        )


        return format_results(
            result,
            source,
            destination
        )


    # ========================================================
    # 3. BOOKING
    # ========================================================

    if (
        "book" in text_lower
        or "booking" in text_lower
        or "ticket" in text_lower
    ):

        return (
            "I can help you find trains "
            "for your journey. Please tell "
            "me your departure station and "
            "destination."
        )


    # ========================================================
    # 4. PNR
    # ========================================================

    if "pnr" in text_lower:

        return (
            "PNR status requires a live "
            "railway service. Please provide "
            "your PNR number."
        )


    # ========================================================
    # 5. FARE
    # ========================================================

    if (
        "fare" in text_lower
        or "price" in text_lower
        or "cost" in text_lower
    ):

        return (
            "Fare information requires a "
            "live railway service. Please "
            "provide the train number and "
            "travel class."
        )


    # ========================================================
    # 6. LIVE STATUS
    # ========================================================

    if (
        "live status" in text_lower
        or "running status" in text_lower
        or "where is my train" in text_lower
    ):

        return (
            "Live train running status "
            "requires a live railway service. "
            "Please provide the train number."
        )


    # ========================================================
    # 7. TIMING
    # ========================================================

    if (
        "timing" in text_lower
        or "time" in text_lower
        or "schedule" in text_lower
    ):

        return (
            "Please provide the train number "
            "or route. For example, say: "
            "show trains from Delhi to Mumbai."
        )


    # ========================================================
    # 8. FALLBACK
    # ========================================================

    return (
        "I could not understand the railway "
        "query. You can ask about a train "
        "number, route, timing, or running days."
    )


# ============================================================
# TEXT TO SPEECH
# ============================================================

def text_to_speech(
    text,
    language="en"
):

    audio_file = tempfile.NamedTemporaryFile(
        delete=False,
        suffix=".mp3"
    )

    audio_file.close()


    try:

        tts = gTTS(
            text=text,
            lang=language
        )


        tts.save(
            audio_file.name
        )


        return audio_file.name


    except Exception as e:

        st.error(
            f"Audio generation failed: {e}"
        )

        return None


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.header(
    "📊 Train Database"
)

st.sidebar.write(
    f"Total records: **{len(trains):,}**"
)

st.sidebar.write(
    "Available columns:"
)

for column in trains.columns:

    st.sidebar.write(
        f"• {column}"
    )


# ============================================================
# CSV INFORMATION
# ============================================================

with st.expander(
    "🔧 CSV Information"
):

    st.write(
        "Detected columns:"
    )

    st.write(
        list(trains.columns)
    )

    st.write(
        "Train number column:",
        TRAIN_NO_COL
    )

    st.write(
        "Train name column:",
        TRAIN_NAME_COL
    )

    st.write(
        "Source column:",
        SOURCE_COL
    )

    st.write(
        "Destination column:",
        DESTINATION_COL
    )

    st.write(
        "Departure column:",
        DEPARTURE_COL
    )

    st.write(
        "Arrival column:",
        ARRIVAL_COL
    )

    st.write(
        "Running days column:",
        RUNS_COL
    )


# ============================================================
# VOICE INPUT
# ============================================================

st.subheader(
    "🎙️ Record or Upload Your Query"
)

audio = st.audio_input(
    "Speak your railway query"
)


# ============================================================
# PROCESS AUDIO
# ============================================================

if audio is not None:

    st.audio(audio)


    with st.spinner(
        "🎧 Processing your voice..."
    ):

        # ----------------------------------------------------
        # Save audio
        # ----------------------------------------------------

        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=".wav"
        ) as temp_audio:

            temp_audio.write(
                audio.getvalue()
            )

            audio_path = temp_audio.name


        try:

            # ------------------------------------------------
            # Whisper
            # ------------------------------------------------

            result = model.transcribe(
                audio_path,
                fp16=False
            )


            text = result[
                "text"
            ].strip()


            detected_language = result.get(
                "language",
                "unknown"
            ).upper()


            # ------------------------------------------------
            # Display transcription
            # ------------------------------------------------

            st.success(
                "Audio processed successfully!"
            )


            st.markdown(
                f"**Detected Language:** "
                f"`{detected_language}`"
            )


            st.markdown(
                f"**You said:** {text}"
            )


            # ------------------------------------------------
            # Process query
            # ------------------------------------------------

            response = process_railway_query(
                text
            )


            # ------------------------------------------------
            # Display response
            # ------------------------------------------------

            st.subheader(
                "🚆 RailVoice Response"
            )


            st.write(
                response
            )


            # ------------------------------------------------
            # Generate audio response
            # ------------------------------------------------

            response_audio = text_to_speech(
                response,
                "en"
            )


            if response_audio:

                st.subheader(
                    "🔊 Audio Response"
                )


                st.audio(
                    response_audio,
                    format="audio/mp3"
                )


        except Exception as e:

            st.error(
                f"❌ Error processing audio: {e}"
            )


        finally:

            if os.path.exists(
                audio_path
            ):

                os.remove(
                    audio_path
                ); 