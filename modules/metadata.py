"""EXIF / metadata forensic analysis."""
from PIL import Image
from PIL.ExifTags import TAGS

# Software / tool signatures commonly left behind by editing or
# generative-AI pipelines. Presence doesn't prove manipulation on its
# own, but it's a legitimate forensic flag worth surfacing to an examiner.
SUSPICIOUS_SOFTWARE_KEYWORDS = [
    "photoshop", "gimp", "lightroom", "affinity",
    "stable diffusion", "midjourney", "dall-e", "dalle", "comfyui",
    "automatic1111", "sdxl", "diffusers", "firefly", "leonardo.ai",
    "playground ai", "ideogram", "flux", "nightcafe", "artbreeder",
    "canva", "runwayml", "krea", "night cafe",
]


def extract_metadata(image: Image.Image) -> dict:
    result = {
        "has_exif": False,
        "raw": {},
        "camera_make": None,
        "camera_model": None,
        "software": None,
        "datetime_original": None,
        "gps_present": False,
        "risk_flags": [],
    }

    try:
        exif_data = image.getexif()
        # Pillow's getexif() misses many tags on some formats; try the
        # legacy private accessor as a fallback for JPEGs.
        if not exif_data and hasattr(image, "_getexif"):
            exif_data = image._getexif()
    except Exception:
        exif_data = None

    if not exif_data:
        result["risk_flags"].append(
            "No EXIF metadata found in this file. Genuine, unedited camera "
            "and phone photos almost always carry EXIF data. Its complete "
            "absence is common in AI-generated images, screenshots, and "
            "images that were stripped or re-saved through another tool — "
            "though it is also common after routine social-media re-uploads."
        )
        return result

    result["has_exif"] = True
    for tag_id, value in dict(exif_data).items():
        tag = TAGS.get(tag_id, str(tag_id))
        result["raw"][str(tag)] = str(value)
        if tag == "Make":
            result["camera_make"] = str(value).strip()
        elif tag == "Model":
            result["camera_model"] = str(value).strip()
        elif tag == "Software":
            result["software"] = str(value).strip()
        elif tag == "DateTimeOriginal":
            result["datetime_original"] = str(value)
        elif tag == "GPSInfo":
            result["gps_present"] = True

    if result["software"]:
        sw_lower = result["software"].lower()
        for kw in SUSPICIOUS_SOFTWARE_KEYWORDS:
            if kw in sw_lower:
                result["risk_flags"].append(
                    f"Software tag reads '{result['software']}', which is "
                    "consistent with AI image generation or significant "
                    "post-processing."
                )
                break

    if not result["camera_make"] and not result["camera_model"]:
        result["risk_flags"].append(
            "No camera make/model recorded — unusual for a photo taken "
            "directly by a camera or smartphone."
        )

    if not result["datetime_original"]:
        result["risk_flags"].append(
            "No original capture timestamp recorded."
        )

    return result
