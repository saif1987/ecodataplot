import os
import requests
from PIL import Image, ImageDraw, ImageOps
from io import BytesIO

# --- Configuration ---

# Match country names used in your main script/data
COUNTRY_ISO_MAP = {
    'China': 'cn',
    'India': 'in',
    'Indonesia': 'id',
    'Pakistan': 'pk',
    'Bangladesh': 'bd',
    'Japan': 'jp',
    'Philippines': 'ph',
    'Viet Nam': 'vn',
    'Iran, Islamic Rep.': 'ir',
    'Turkiye': 'tr', # Note: flagcdn uses 'tr' for Turkey/Turkiye
    'Thailand': 'th',
    'Myanmar': 'mm',
    'Korea, Rep.': 'kr' # Note: flagcdn uses 'kr' for South Korea
    # Add other countries from your ASIAN_COUNTRIES list if needed
}

ICON_SIZE = (64, 64) # Desired icon size in pixels
OUTPUT_DIR = "../icons" # Relative path to save icons (adjust if needed)
FLAG_CDN_URL_TEMPLATE = "https://flagcdn.com/w160/{iso_code}.png" # w160 provides 160px wide images

def create_rounded_image(img: Image.Image, size: tuple[int, int]) -> Image.Image:
    """Creates a rounded version of the input image."""
    # Create a mask
    mask = Image.new('L', size, 0)
    draw = ImageDraw.Draw(mask)
    draw.ellipse((0, 0) + size, fill=255)

    # Ensure image is RGBA for transparency
    img = img.convert("RGBA")
    # Apply mask
    output = ImageOps.fit(img, mask.size, centering=(0.5, 0.5))
    output.putalpha(mask)

    return output.resize(size, Image.LANCZOS) # Use older constant for compatibility

def fetch_and_process_flags():
    """Downloads flags, rounds them, and saves them."""
    if not os.path.exists(OUTPUT_DIR):
        print(f"Creating output directory: {OUTPUT_DIR}")
        os.makedirs(OUTPUT_DIR)

    session = requests.Session() # Use a session for potential connection reuse

    for country_name, iso_code in COUNTRY_ISO_MAP.items():
        flag_url = FLAG_CDN_URL_TEMPLATE.format(iso_code=iso_code.lower())
        output_filename = os.path.join(OUTPUT_DIR, f"{country_name}.png")

        if os.path.exists(output_filename):
            print(f"Skipping {country_name}, icon already exists.")
            continue

        print(f"Fetching flag for {country_name} ({iso_code})...")
        try:
            response = session.get(flag_url, timeout=10)
            response.raise_for_status() # Raise an error for bad status codes

            img_data = BytesIO(response.content)
            img = Image.open(img_data)

            print(f"Processing and saving icon for {country_name}...")
            rounded_img = create_rounded_image(img, ICON_SIZE)
            rounded_img.save(output_filename, "PNG")

        except requests.exceptions.RequestException as e:
            print(f"  Error fetching flag for {country_name}: {e}")
        except Exception as e:
            print(f"  Error processing image for {country_name}: {e}")

if __name__ == "__main__":
    fetch_and_process_flags()
    print("Flag fetching process complete.")