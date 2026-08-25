from flask import Flask, render_template, request, jsonify
from flask import redirect

from dotenv import load_dotenv

import requests
import json
import os
load_dotenv()

import time
import random

from astropy.io import fits 
from astropy.stats import sigma_clipped_stats
from astropy.io import fits
from astropy.wcs import WCS
from astropy.io import ascii
import astropy.units as u
from astropy.time import Time
from astropy.coordinates import SkyCoord, Angle
from astropy.wcs.utils import skycoord_to_pixel
from astroquery.simbad import Simbad
from astropy.visualization import ImageNormalize, ZScaleInterval, AsinhStretch


from photutils.aperture import CircularAperture, CircularAnnulus, aperture_photometry
from photutils.centroids import centroid_com

from matplotlib.patches import Circle
import matplotlib.pyplot as plt 
import numpy as np
import pyvo
import pandas as pd
import matplotlib
matplotlib.use("Agg")

from photutils.detection import DAOStarFinder

os.makedirs("static", exist_ok=True)


global status_message
global last_jd
status_message = "Waiting for user input..."
error_message = None
last_g = None
last_r = None
last_err_g = None
last_err_r = None
last_ra = None
last_dec = None
last_jd = None
last_ra_hms = None
last_dec_dms = None
last_target_name = None

progress_value = 0
progress_message = "Waiting for user input..."


def update_progress(value, message):
    global progress_value, progress_message
    if progress_value == 0 or progress_value == 100:
        progress_value = value
        progress_message = message
    else:
        progress_value += value
        progress_message = message

#=========================================
# Working with Astronometry and APASS APIs 
#=========================================

def login_to_astrometry(api_key: str) -> str:
	url = 'http://nova.astrometry.net/api/login'
	payload = {'request-json': json.dumps({"apikey": api_key})}
	response = requests.post(url, data=payload)
	try:
		session_key = response.json().get('session')
	except Exception:
		session_key = None
	return session_key


def upload_fits_file(file_path, session_token, url="http://nova.astrometry.net/api/upload"):

    request_json = {
        "publicly_visible": "y",
        "allow_modifications": "d",
        "session": session_token,
        "allow_commercial_use": "d"
    }

    with open(file_path, "rb") as f:
        files = {
            "request-json": (
                None,
                json.dumps(request_json),
                "text/plain"
            ),
            "file": (
                file_path,
                f,
                "application/octet-stream"
            )
        }
        response = requests.post(url, files=files)

    # print("Status code:", response.status_code)
    # print("Response:", response.text)
    try:
        subid = response.json().get("subid")
    except Exception:
        subid = None
    return subid

def safe_get_json(url):
    """
    Safely GET a URL and return JSON if possible.
    If the response is not JSON, print an error and return None.
    """
    try:
        resp = requests.get(url, timeout=10)
    except Exception as e:
        print(f"Network error while requesting {url}: {e}")
        return None

    try:
        return resp.json()
    except ValueError:
        print(f"Astrometry.net returned non‑JSON response for {url}:")
        status_message = f"Astrometry.net returned non‑JSON response for {url}: {resp.text[:500]} ... Please try again later."
        print(resp.text[:500])  # print first 500 chars for debugging
        return None

def wait_for_job(sub_id, timeout=180):
    url = f"http://nova.astrometry.net/api/submissions/{sub_id}"

    for i in range(timeout):
        r = safe_get_json(url)
        if r is None:
            print("Error: Could not retrieve submission status.")
            return None

        jobs = r.get("jobs", [])
        if jobs and jobs[0] is not None:
            print(f"Job found: {jobs[0]}")
            return jobs[0]

        print(f"Waiting for job... {i}")
        time.sleep(2)

    raise TimeoutError("Job did not appear in time.")


def wait_for_calibration(job_id, timeout=180):
    url = f"http://nova.astrometry.net/api/jobs/{job_id}/calibration/"

    for _ in range(timeout):
        r = safe_get_json(url)
        if r is None:
            print("Error: Could not retrieve calibration status.")
            return None

        if "ra" in r:
            return r

        time.sleep(2)

    return None


def apply_calibration_to_fits(input_fits, output_fits, job_id):

    global status_message

    """
    Safely apply the exact WCS header from Astrometry.net by embedding
    the downloaded header into a valid FITS HDU before parsing.
    """

    # 1. Download WCS header text
    wcs_url = f"http://nova.astrometry.net/wcs_file/{job_id}"
    r = requests.get(wcs_url)
    r.raise_for_status()
    wcs_header_text = r.text

    # 2. Convert raw header text into a Header object manually
    #    (no parsing yet — just splitting lines)
    header_lines = wcs_header_text.split("\n")

    # 3. Build a minimal valid FITS header
    hdr = fits.Header()

    # Required FITS cards
    hdr["SIMPLE"] = True
    hdr["BITPIX"] = -32
    hdr["NAXIS"] = 0
    hdr["EXTEND"] = True

    # Append all WCS cards
    for line in header_lines:
        line = line.strip()
        if len(line) >= 8 and "=" in line[:10]:
            key = line[:8].strip()
            val_comment = line[9:].strip()
            try:
                hdr[key] = val_comment
            except Exception:
                pass  # ignore malformed cards safely

    # 4. Now Astropy can parse this header safely
    wcs = WCS(hdr)

    # 5. Merge WCS into your real FITS file
    with fits.open(input_fits) as hdul:
        data = hdul[0].data
        real_hdr = hdul[0].header

        for key, val in hdr.items():
            real_hdr[key] = val

        fits.writeto(output_fits, data, real_hdr, overwrite=True)

    print(f"Exact Astrometry.net WCS written to {output_fits}")
    status_message = f"WCS written to {output_fits}..."

    return wcs


def query_apass_to_csv(ra_center, dec_center, radius_deg, output_csv="apass_subset.csv"):
    global status_message

    """
    Query APASS DR10 catalog using GAVO TAP and save results to a CSV file.
    Args:
        ra_center (float): Right Ascension of center (degrees)
        dec_center (float): Declination of center (degrees)
        radius_deg (float): Search radius (degrees)
        output_csv (str): Output CSV filename
    """
    service = pyvo.dal.TAPService("https://dc.g-vo.org/tap")
    query = f"""
    SELECT * 
    FROM apass.dr10
    WHERE 1 = CONTAINS(
        POINT('ICRS', ra, dec),
        CIRCLE('ICRS', {ra_center}, {dec_center}, {radius_deg})
    )
    """
    result = service.search(query)
    df = result.to_table().to_pandas()
    df = df.dropna(subset=["ra", "dec", "mag_g", "mag_r"])
    df.to_csv(output_csv, index=False)
    print(f"Saved {output_csv}")
    num_rows = len(df)
    status_message = f"Saved csv file as {output_csv}..."
    status_message = f"Image processing is done! Please wait for Photometry to finish..."

    return num_rows - 2


def full_calibration_with_subid(image, wcs_image_name, subid_key):
    global status_message

    # 1. Login to Astrometry.net
    api_key = os.environ.get("ASTRO_LOGIN")
    session_key = login_to_astrometry(api_key)
    print("Session:", session_key)
    update_progress(5, "Logging into Astrometry.net...")

    
    #status_message = f"Session: {session_key}"
    status_message = f"Session key achieved...please wait for submission ID"
    update_progress(5, "Uploading FITS file to Astrometry.net...")


    # 2. Upload image if no subid_key provided
    if subid_key is None:
        subid_key = upload_fits_file(image, session_key)
        print("Submission ID:", subid_key)
        #status_message = f"Submission ID: {subid_key}"
        status_message = f"Submission ID achieved...please wait for job ID"
        update_progress(5, "Waiting for sub ID from Astrometry.net...")

    else:
        subid_key = subid_key
        print("Submission ID:", subid_key)
        #status_message = f"Submission ID: {subid_key}"
        status_message = f"Submission ID achieved...please wait for job ID"

    # 3. Wait for job to finish
    # 3. Wait for job to finish
    try:
        job_id = wait_for_job(subid_key)
        update_progress(5, "Waiting for job ID from Astrometry.net...")
    except TimeoutError:
        print("Timeout waiting for job.")
        status_message = "Error: Astrometry.net did not return a job ID in time."
        return None

    print("Job ID:", job_id)
    #status_message = f"Job ID: {job_id}"
    status_message = f"Job ID achieved...please wait for astrometry results"
    update_progress(5, "Job ID received. Waiting for astrometric calibration...")


    # 4. Retrieve calibration results
    astro_results = wait_for_calibration(job_id)
    print("Astrometry Results:", astro_results)
    #status_message = f"Astrometry Results: {astro_results}"
    status_message = f"Astrometry Results achieved...please wait for WCS application"
    update_progress(5, "Astrometry calibration received. Applying WCS...")


    # 5. Apply WCS to the green image
    image = apply_calibration_to_fits(image, wcs_image_name, astro_results)

    # 6. Query APASS around the solved coordinates
    ra = astro_results["ra"]
    dec = astro_results["dec"]
    radius = round(astro_results["radius"] * 0.6, 1)
    update_progress(5, "WCS applied. Querying APASS DR10 catalog...")


    num_rows = query_apass_to_csv(ra, dec, radius, "apass_subset.csv")
    status_message = f"Image processing is done! Please wait for Photometry to finish..."
    update_progress(5, "APASS catalog loaded. Preparing for photometry...")

    
    return num_rows

#=========================================
# Gathering specific information from APIs 
#=========================================

def full_calibration(image, wcs_image_name):
    global status_message

    # 1. Login to Astrometry.net
    api_key = os.environ.get("ASTRO_LOGIN")
    session_key = login_to_astrometry(api_key)
    print("Session:", session_key)

    status_message = (
        "Session key achieved...please wait for submission ID..."
        "please be patient as this will be the longest step!"
    )
    update_progress(5, "Logging into Astrometry.net...")

    # 2. Upload image
    subid_key = upload_fits_file(image, session_key)
    print("Submission ID:", subid_key)

    status_message = "Submission ID achieved...please wait for job ID"
    update_progress(5, "Uploading FITS file to Astrometry.net...")

    # 3. Wait for job to finish
    job_id = wait_for_job(subid_key)
    print("Job ID:", job_id)

    status_message = "Job ID achieved...please wait for astrometry results"
    update_progress(5, "Waiting for job ID from Astrometry.net...")

    # 4. Retrieve calibration results
    astro_results = wait_for_calibration(job_id)
    print("Astrometry Results:", astro_results)

    status_message = "Astrometry Results achieved...please wait for WCS application"
    update_progress(5, "Job ID received. Waiting for astrometric calibration...")

    # 5. Apply WCS
    image = apply_calibration_to_fits(image, wcs_image_name, astro_results)

    # 6. Query APASS
    ra = astro_results["ra"]
    dec = astro_results["dec"]
    radius = round(astro_results["radius"] * 0.6, 1)

    update_progress(5, "Astrometry calibration received. Applying WCS...")
    update_progress(5, "WCS applied. Querying APASS DR10 catalog...")

    num_rows = query_apass_to_csv(ra, dec, radius, "apass_subset.csv")

    status_message = "Image Processing is done! Please wait for Photometry to finish..."
    update_progress(5, "APASS catalog loaded. Preparing for photometry...")

    return num_rows





#======================================
# Functions used for Target Photometry 
#======================================

def detect_stars(image, fwhm=3.0, sigma=3.0, threshold=5.0):
    data = fits.getdata(image)
    mean, median, std = sigma_clipped_stats(data, sigma=sigma)
    daofind = DAOStarFinder(fwhm=fwhm, threshold=threshold * std)
    sources = daofind(data - median)
    return sources


def match_detected_to_apass(det_x, det_y, wcs, apass_csv, max_arcsec=1.2):

    # Convert detections to RA/Dec
    det_ra, det_dec = wcs.all_pix2world(det_x, det_y, 1)
    det_coords = SkyCoord(det_ra*u.deg, det_dec*u.deg)

    # Load APASS
    cat = pd.read_csv(apass_csv)
    cat_coords = SkyCoord(cat["ra"].values*u.deg,
                          cat["dec"].values*u.deg)

    # Match
    idx, sep2d, _ = det_coords.match_to_catalog_sky(cat_coords)

    # Keep only very close matches
    good = sep2d < (max_arcsec * u.arcsec)

    matched = {
        "det_x": det_x[good],
        "det_y": det_y[good],
        "cat_index": idx[good],
        "sep_arcsec": sep2d[good].arcsec,
        "g": cat["mag_g"].values[idx[good]],
        "r": cat["mag_r"].values[idx[good]]
    }

    return matched


def flux(x, y, radius, image):
    """
    Perform circular aperture photometry with an annulus background.
    x, y: arrays of centroid positions (floats)
    radius: aperture radius in pixels
    image: FITS filename
    """
    data = fits.getdata(image)

    # Convert to Nx2 array of positions
    positions = np.column_stack((x, y))

    # Circular aperture
    aperture = CircularAperture(positions, r=radius)

    # Background annulus (3 px wide)
    annulus = CircularAnnulus(positions, r_in=radius+3, r_out=radius+6)

    # Perform photometry
    aper_phot = aperture_photometry(data, aperture)
    ann_phot = aperture_photometry(data, annulus)

    # Compute background per pixel
    bkg_mean = ann_phot['aperture_sum'] / annulus.area

    # Subtract background from aperture flux
    final_flux = aper_phot['aperture_sum'] - bkg_mean * aperture.area

    # Replace negative flux with zero
    final_flux = np.where(final_flux > 0, final_flux, 0)

    return np.array(final_flux)


def target_flux(x, y, radius, image):
    """
    Same as flux(), but for a single target star.
    """
    data = fits.getdata(image)

    position = np.array([[x, y]])

    aperture = CircularAperture(position, r=radius)
    annulus = CircularAnnulus(position, r_in=radius+3, r_out=radius+6)

    aper_phot = aperture_photometry(data, aperture)
    ann_phot = aperture_photometry(data, annulus)

    bkg_mean = ann_phot['aperture_sum'] / annulus.area
    final_flux = aper_phot['aperture_sum'] - bkg_mean * aperture.area

    return np.array([max(final_flux[0], 0)])



def show_target_cutout(
    green_fits,
    red_fits,
    wcs_green,
    wcs_red,
    target_ra,
    target_dec,
    radius=8,
    search_arcsec=4,
    output_combined="static/target_cutout_combined.png"
):
    """
    Produce TWO target cutouts (green + red) and combine them into one figure.
    Returns:
        (cx, cy, star_name, status_message, output_combined)
    """

    update_progress(5, "Locating target star in green image...")

    # -----------------------------
    # Load images
    # -----------------------------
    data_g = fits.getdata(green_fits)
    data_r = fits.getdata(red_fits)
    ny, nx = data_g.shape

    # -----------------------------
    # Convert RA/Dec → pixel (green)
    # -----------------------------
    target_coord = SkyCoord(target_ra*u.deg, target_dec*u.deg)
    tx_g, ty_g = skycoord_to_pixel(target_coord, wcs_green)

    # -----------------------------
    # Detect stars in GREEN image
    # -----------------------------
    sources = detect_stars(green_fits, threshold=3.0)
    det_x = np.array(sources["xcentroid"])
    det_y = np.array(sources["ycentroid"])

    det_ra, det_dec = wcs_green.all_pix2world(det_x, det_y, 1)
    det_coords = SkyCoord(det_ra*u.deg, det_dec*u.deg)

    # -----------------------------
    # Nearest detected star
    # -----------------------------
    sep = det_coords.separation(target_coord)
    min_sep = sep.min().arcsec
    idx = sep.argmin()

    # -----------------------------
    # If no star found near RA/Dec
    # -----------------------------
    if min_sep > search_arcsec:
        msg = (
            f"No detected star within {search_arcsec} arcsec. "
            "Showing cutouts centered on the requested RA/Dec."
        )

        cx_g, cy_g = tx_g, ty_g
    else:
        cx_g, cy_g = det_x[idx], det_y[idx]

    # -----------------------------
    # Refine centroid in GREEN cutout
    # -----------------------------
    half = int(radius * 4)
    x1, x2 = int(cx_g - half), int(cx_g + half)
    y1, y2 = int(cy_g - half), int(cy_g + half)

    x1 = max(x1, 0); y1 = max(y1, 0)
    x2 = min(x2, nx); y2 = min(y2, ny)

    cut_g = data_g[y1:y2, x1:x2]

    cy_local, cx_local = centroid_com(cut_g)
    cx_g = x1 + cx_local
    cy_g = y1 + cy_local

    # -----------------------------
    # Project centroid into RED image
    # -----------------------------
    cx_r, cy_r = wcs_red.all_world2pix(
        *wcs_green.all_pix2world(cx_g, cy_g, 1),
        1
    )

    # -----------------------------
    # Make RED cutout
    # -----------------------------
    x1r, x2r = int(cx_r - half), int(cx_r + half)
    y1r, y2r = int(cy_r - half), int(cy_r + half)

    x1r = max(x1r, 0); y1r = max(y1r, 0)
    x2r = min(x2r, nx); y2r = min(y2r, ny)

    cut_r = data_r[y1r:y2r, x1r:x2r]

    # -----------------------------
    # Identify star name (same as before)
    # -----------------------------
    star_name = None
    try:
        Simbad.add_votable_fields("ids", "id_gaia_dr3")
        result = Simbad.query_region(target_coord, radius=Angle(5, "arcsec"))
        if result is not None:
            gaia_id = result["ID_GAIA_DR3"][0]
            if gaia_id not in (None, "", b"", " "):
                star_name = f"Gaia DR3 {gaia_id}"
            else:
                star_name = result["MAIN_ID"][0].decode("utf-8")
    except Exception:
        pass

    if star_name is None:
        try:
            from astroquery.vizier import Vizier
            Vizier.ROW_LIMIT = 1
            result = Vizier.query_region(
                target_coord,
                radius=Angle(5, "arcsec"),
                catalog="I/355/gaiadr3"
            )
            if len(result) > 0:
                star_name = f"Gaia DR3 {result[0]['Source'][0]}"
        except Exception:
            pass

    if star_name is None:
        try:
            apass_df = pd.read_csv("apass_subset.csv")
            apass_coords = SkyCoord(apass_df["ra"]*u.deg, apass_df["dec"]*u.deg)
            idx_apass = apass_coords.separation(target_coord).argmin()
            star_name = f"APASS DR10 {idx_apass}"
        except Exception:
            star_name = "Unknown"

    msg = f"Target star found at {min_sep:.2f} arcsec from input coordinates."

    # -----------------------------
    # Create combined figure
    # -----------------------------
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    # GREEN
    axes[0].imshow(
        cut_g, cmap="gray", origin="lower",
        vmin=np.percentile(cut_g, 5),
        vmax=np.percentile(cut_g, 99)
    )
    axes[0].add_patch(Circle((cx_g - x1, cy_g - y1), radius, edgecolor='yellow', facecolor='none'))
    axes[0].add_patch(Circle((cx_g - x1, cy_g - y1), radius+3, edgecolor='cyan', facecolor='none'))
    axes[0].add_patch(Circle((cx_g - x1, cy_g - y1), radius+6, edgecolor='cyan', facecolor='none'))
    axes[0].set_title("Green Cutout")
    axes[0].set_xticks([]); axes[0].set_yticks([])

    # RED
    axes[1].imshow(
        cut_r, cmap="gray", origin="lower",
        vmin=np.percentile(cut_r, 5),
        vmax=np.percentile(cut_r, 99)
    )
    axes[1].add_patch(Circle((cx_r - x1r, cy_r - y1r), radius, edgecolor='yellow', facecolor='none'))
    axes[1].add_patch(Circle((cx_r - x1r, cy_r - y1r), radius+3, edgecolor='cyan', facecolor='none'))
    axes[1].add_patch(Circle((cx_r - x1r, cy_r - y1r), radius+6, edgecolor='cyan', facecolor='none'))
    axes[1].set_title("Red Cutout")
    axes[1].set_xticks([]); axes[1].set_yticks([])

    plt.tight_layout()
    plt.savefig(output_combined, dpi=150, bbox_inches="tight")
    plt.close()

    return cx_g, cy_g, star_name, msg, output_combined


def get_image_center(fits_file):
    with fits.open(fits_file) as hdul:
        data = hdul[0].data
        w = WCS(hdul[0].header)

    ny, nx = data.shape
    cx_pix = nx / 2
    cy_pix = ny / 2

    ra_center, dec_center = w.all_pix2world(cx_pix, cy_pix, 1)

    # ⭐ Convert NumPy arrays → Python floats
    ra_center = float(ra_center)
    dec_center = float(dec_center)

    return ra_center, dec_center



def lsrl(x, y):
    x = np.array(x)
    y = np.array(y)

    if np.std(x) < 1e-3 or np.std(y) < 1e-3:
        raise ValueError("Not enough color variation for calibration.")

    m, b = np.polyfit(x, y, 1)
    sd = np.std(y - (m*x + b))
    return m, b, sd



def magnitudes(csv_file, green_image, red_image, n, RA, DEC):
    global status_message
    global last_target_name



    data = ascii.read(csv_file, format='csv')
    hdul_g = fits.open(green_image)
    hdul_r = fits.open(red_image)

    with fits.open(green_image) as hdul:
        image_data_g = hdul[0].data

    with fits.open(red_image) as hdul:
        image_data_r = hdul[0].data

    w_g = WCS(hdul_g[0].header)
    w_r = WCS(hdul_r[0].header)


    cx, cy, star_name, msg, cutout_path = show_target_cutout(
        "wcs_green_solution.fits",
        "wcs_red_solution.fits",
        w_g,
        w_r,
        RA,
        DEC,
        radius=8,
        search_arcsec=4,
        output_combined="static/target_cutout.png"
    )


    update_progress(2, "Beginning photometry pipeline...")


    col_length = len(data['ra'])

    # Ensure n never exceeds available stars
    n = min(n, col_length)

    calibration_num = random.sample(range(col_length), n)
    calibration_num = sorted(calibration_num)

    ra_list = np.array([])
    dec_list = np.array([])

    x_pixel_g = np.array([])
    y_pixel_g = np.array([])
    x_pixel_r = np.array([])
    y_pixel_r = np.array([])
    mag_g = np.array([])
    mag_r = np.array([])
    g = np.array([])
    r = np.array([])

    j = 0

    for i in range(col_length):
        if j < len(calibration_num) and i == calibration_num[j]:
            ra = data['ra'][i]
            dec = data['dec'][i]
            x_g, y_g = w_g.all_world2pix(ra, dec, 1)
            x_r, y_r = w_r.all_world2pix(ra, dec, 1)
            ra_list = np.append(ra_list, ra)
            dec_list = np.append(dec_list, dec)
            x_pixel_g = np.append(x_pixel_g, x_g)
            y_pixel_g = np.append(y_pixel_g, y_g)
            x_pixel_r = np.append(x_pixel_r, x_r)
            y_pixel_r = np.append(y_pixel_r, y_r)
            g = np.append(g, data['mag_g'][i])
            r = np.append(r, data['mag_r'][i])

            j += 1

    update_progress(1, "Detecting stars in green and red images...")

 # === GREEN WCS CHECK ===

    # Unified higher-contrast percentile clamp
    pmin, pmax = np.percentile(image_data_g, [20, 80])

    # ZScale limits
    z = ZScaleInterval()
    zmin, zmax = z.get_limits(image_data_g)

    # Blended limits (same for both images)
    vmin = max(zmin, pmin)
    vmax = min(zmax, pmax)

    norm_g = ImageNormalize(
        image_data_g,
        vmin=vmin,
        vmax=vmax,
        stretch=AsinhStretch()
    )

    fig = plt.figure(figsize=(10,8))
    ax = plt.subplot(projection=w_g)

    ax.coords[0].set_axislabel("Right Ascension (RA)")

    ax.coords[1].set_axislabel_position('l') 
    ax.coords[1].set_ticklabel_position('l')
    ax.imshow(image_data_g, cmap="gray", origin="lower", norm=norm_g)

    # Compute image center for display 
    ra_center, dec_center = get_image_center("wcs_green_solution.fits")



  # Calibration Stars marked
    for xg, yg in zip(x_pixel_g, y_pixel_g):
        ax.add_patch(plt.Circle(
            (xg, yg),
            radius=20,                 # slightly bigger
            edgecolor='#5A0000',       # darker red
            facecolor='none',
            linewidth=0.9,
            alpha=0.6
        ))

    # Target star unchanged
    target_g_x, target_g_y = w_g.all_world2pix(RA, DEC, 1)
    ax.add_patch(plt.Circle((target_g_x, target_g_y), 25,
                            edgecolor='black', facecolor='none', linewidth=0.8))
    ax.text(target_g_x + 10, target_g_y + 10, "Target", color='black')

    plt.title("GREEN WCS Check: APASS stars + Target")
    plt.savefig("static/green_wcs_check.png", dpi=150, bbox_inches="tight")
    plt.close()



    # === RED WCS CHECK ===

    # Use the SAME contrast settings for red image
    pmin, pmax = np.percentile(image_data_r, [20, 80])
    z = ZScaleInterval()
    zmin, zmax = z.get_limits(image_data_r)
    vmin = max(zmin, pmin)
    vmax = min(zmax, pmax)

    norm_r = ImageNormalize(
        image_data_r,
        vmin=vmin,
        vmax=vmax,
        stretch=AsinhStretch()
    )

    fig = plt.figure(figsize=(10,8))
    ax = plt.subplot(projection=w_r)
    ax.coords[0].set_axislabel("Right Ascension (RA)")
    ax.coords[1].set_axislabel_position('l') 
    ax.coords[1].set_ticklabel_position('l')
    ax.imshow(image_data_r, cmap="gray", origin="lower", norm=norm_r)

    # Calibration stars: slightly bigger, opacity 0.6, darker blue-green
    for xr, yr in zip(x_pixel_r, y_pixel_r):
        ax.add_patch(plt.Circle(
            (xr, yr),
            radius=20,                 # slightly bigger
            edgecolor='#004F4F',       # darker cyan/teal
            facecolor='none',
            linewidth=0.9,
            alpha=0.6
        ))

    # Target star unchanged
    target_r_x, target_r_y = w_r.all_world2pix(RA, DEC, 1)
    ax.add_patch(plt.Circle((target_r_x, target_r_y), 25,
                            edgecolor='black', facecolor='none', linewidth=0.8))
    ax.text(target_r_x + 10, target_r_y + 10, "Target", color='black')

    plt.title("RED WCS Check: APASS stars + Target")
    plt.savefig("static/red_wcs_check.png", dpi=150, bbox_inches="tight")
    plt.close()



    

    red_wcs_path = "static/red_wcs_check.png"
    green_wcs_path = "static/green_wcs_check.png"


    # === IMAGE SIZE FOR APASS FILTERING ===
    ny, nx = image_data_g.shape


    update_progress(1, "Matching detected stars to APASS catalog...")


    # === 1. Remove APASS stars outside the image ===

    inside = (
        (x_pixel_g >= 0) & (x_pixel_g < nx) &
        (y_pixel_g >= 0) & (y_pixel_g < ny)
    )

    x_pixel_g = x_pixel_g[inside]
    y_pixel_g = y_pixel_g[inside]
    x_pixel_r = x_pixel_r[inside]
    y_pixel_r = y_pixel_r[inside]
    g = g[inside]
    r = r[inside]


    # ============================
    #   GREEN IMAGE (MASTER BAND)
    # ============================

    # 2.a. Detect stars in the green image
    sources = detect_stars(green_image, threshold=3.0)
    det_x_g = np.array(sources['xcentroid'])
    det_y_g = np.array(sources['ycentroid'])

    # 2.b. Match detected stars to APASS using GREEN WCS
    matched = match_detected_to_apass(det_x_g, det_y_g, w_g, "apass_subset.csv")

    # 2.c. Load APASS catalog
    cat = pd.read_csv("apass_subset.csv")

    # 2.d. Extract RA/Dec of the APASS stars that matched in GREEN
    ra_match  = cat["ra"].values[matched["cat_index"]]
    dec_match = cat["dec"].values[matched["cat_index"]]

    # Pixel positions of these stars in the GREEN image
    x_green = matched["det_x"]
    y_green = matched["det_y"]

    # ============================
    #        RED IMAGE
    # ============================

    # 3. Project the SAME APASS stars into the RED image using RED WCS
    x_red, y_red = w_r.all_world2pix(ra_match, dec_match, 1)


    update_progress(1, "Computing instrumental fluxes...")


        
    green_flux = flux(x_green, y_green, 7, green_image)
    red_flux = flux(x_red, y_red, 7, red_image)


    valid_flux_g = []
    valid_flux_r = []

    mask = (
    (green_flux > 0) &
    np.isfinite(green_flux) &
    (red_flux > 0) &
    np.isfinite(red_flux)
    )   

    valid_flux_g = green_flux[mask]
    valid_flux_r = red_flux[mask]
    g = matched["g"][mask]
    r = matched["r"][mask]

    update_progress(1, "Computing instrumental magnitudes (g, r)...")


    valid_flux_g = np.array(valid_flux_g)
    valid_flux_r = np.array(valid_flux_r)

    inst_g = -2.5 * np.log10(valid_flux_g)
    inst_r = -2.5 * np.log10(valid_flux_r)

    update_progress(1, "Solving color transformation equations...")


    #X1
    inst_g_r = inst_g - inst_r
    
    # Y1 & X2
    st_g_r = g - r
    
    # Y2
    g_offset = g - inst_g



    # ============================================
    # LIMIT TO n CALIBRATION STARS *AFTER MATCHING* ONLY IF USER WISHES TO REDUCE NUMBER OF CALIBRATION STARS
    # ============================================

    total = len(inst_g)
    n = min(n, total)

    chosen = np.random.choice(total, n, replace=False)

    inst_g = inst_g[chosen]
    inst_r = inst_r[chosen]
    inst_g_r = inst_g_r[chosen]
    st_g_r = st_g_r[chosen]
    g_offset = g_offset[chosen]
    g = g[chosen]
    r = r[chosen]

    ##Specifically solving for the target object now.

    target_pixel_g_x, target_pixel_g_y = w_g.all_world2pix(RA, DEC, 1)
    target_pixel_r_x, target_pixel_r_y = w_r.all_world2pix(RA, DEC, 1)

    
  
    status_message = msg
    last_target_name = star_name
    last_target_cutout = cutout_path

    update_progress(1, "Applying color transformation to target star...")

    target_flux_g = target_flux(target_pixel_g_x, target_pixel_g_y, 8, green_image)
    target_flux_r = target_flux(target_pixel_r_x, target_pixel_r_y, 8, red_image)

    target_g_inst_mag = -2.5 * np.log10(target_flux_g)
    target_r_inst_mag = -2.5 * np.log10(target_flux_r)

    m1_b1 = lsrl(inst_g_r, st_g_r)
    m2_b2 = lsrl(st_g_r, g_offset)

    new_std = m1_b1[0]*(inst_g_r) + m1_b1[1]
    new_std_inst = m2_b2[0]*new_std + m2_b2[1] 

    update_progress(1, "Generating diagnostic plots...")

    #================================
    # Plotting out calibration stars and linear regression
    #================================


    plt.figure(figsize=(7,5))
    plt.plot(inst_g_r, new_std, label=f'Tgr = {round(m1_b1[0], 4)} \n Cgr = {round(m1_b1[1], 4)}')
    plt.scatter(inst_g_r, st_g_r)
    plt.xlabel('Instrumental (g-r)')
    plt.ylabel('Standard (g-r)')
    plt.title('Instrumental Color Index vs. Standard Color Index')
    plt.legend()

    color_term_path = "static/object_color_term.png"
    plt.savefig(color_term_path, dpi=150, bbox_inches="tight")
    plt.close()

    
    
    plt.figure(figsize=(7,5))
    plt.plot(new_std, new_std_inst, label=f"Tg = {round(m2_b2[0], 4)}")
    plt.scatter(new_std, g_offset)
    plt.xlabel('Standard (g-r)')
    plt.ylabel('Offset')
    plt.title('Standard Color Index vs. Green Offset')
    plt.legend()

    green_offset_path = "static/object_green_offset.png"
    plt.savefig(green_offset_path, dpi=150, bbox_inches="tight")
    plt.close()




    standard_g_r_target = m1_b1[0]*(target_g_inst_mag - target_r_inst_mag) + m1_b1[1]
    standard_g_target = m2_b2[0]*standard_g_r_target + m2_b2[1] + target_g_inst_mag
    
    standard_r_target = standard_g_target - standard_g_r_target

    # === CHECK FOR NAN TARGET MAGNITUDES ===
    if np.isnan(standard_g_target) or np.isnan(standard_r_target):
        status_message = (
            "The target star could not be measured. "
            "It was either too faint, outside the aperture, or had insufficient flux "
            "for reliable photometric calibration."
        )
        return (
            "N/A",  # standard g
            "N/A",  # standard r
            "N/A",  # error g
            "N/A",  # error r
            None,
            None,
            None,
            None,  
            color_term_path,
            green_offset_path,
            red_wcs_path,
            green_wcs_path,
            last_target_name,
            last_target_cutout
        )

    sigma1 = m1_b1[2]
    sigma2 = m2_b2[2]

    error_g = (sigma1**2 + sigma2**2)**0.5
    error_r = (2*sigma1**2 + sigma2**2)**0.5


    Tgr = m1_b1[0]
    Cgr = m1_b1[1]
    Tg  = m2_b2[0]
    Cg  = m2_b2[1]


    status_message = "Done!"
    update_progress(100, "Photometry complete!")

    return (
        round(standard_g_target[0], 5),
        round(standard_r_target[0], 5),
        round(error_g, 5),
        round(error_r, 5),
        round(Tgr, 4),
        round(Cgr, 4),
        round(Tg, 4),
        round(Cg, 4),
        color_term_path,
        green_offset_path,
        red_wcs_path,
        green_wcs_path,
        last_target_name, 
        last_target_cutout, 
        ra_center,
        dec_center

    )


#======================================
# If user just wants to calibrate the images, use this function
#======================================
def calibrate_image_only(fits_file):

    global status_message

    base = "calibrated"  # fixed base name

    wcs_fits_path = f"static/{base}_wcs.fits"
    png_path      = f"static/{base}_preview.png"


    

    # -----------------------------
    # 1. Login
    # -----------------------------
    update_progress(5, "Logging into Astrometry.net...")
    api_key = os.environ.get("ASTRO_LOGIN")
    session_key = login_to_astrometry(api_key)

    if session_key is None:
        status_message = "Error: Astrometry.net login failed."
        raise RuntimeError("Astrometry.net login failed.")

    status_message = "Session key acquired. Uploading FITS file..."
    update_progress(5, "Session key acquired.")

    # -----------------------------
    # 2. Upload + Solve
    # -----------------------------
    subid = upload_fits_file(fits_file, session_key)
    update_progress(10, "FITS uploaded. Waiting for job ID...")

    job_id = wait_for_job(subid)
    update_progress(10, "Job ID received. Waiting for calibration...")

    # -----------------------------
    # 3. Retrieve FULL calibration metadata
    # -----------------------------
    astro_metadata = wait_for_calibration(job_id)
    update_progress(10, "Calibration received. Applying WCS...")

    # Print everything to console for debugging
    print("\n===== ASTROMETRY.NET METADATA =====")
    for key, value in astro_metadata.items():
        print(f"{key}: {value}")
    print("===================================\n")

    # -----------------------------
    # 4. Apply WCS to FITS
    # -----------------------------
    wcs_obj = apply_calibration_to_fits(fits_file, wcs_fits_path, job_id)
    status_message = "WCS applied successfully."
    update_progress(10, "WCS applied.")

    # -----------------------------
    # 5. Compute image center
    # -----------------------------
    ra_center, dec_center = get_image_center(wcs_fits_path)
    status_message = f"Image center computed: RA={ra_center:.5f}, Dec={dec_center:.5f}"
    update_progress(10, "Computed image center.")

    # -----------------------------
    # 6. Create PNG preview
    # -----------------------------
    png_path = "static/wcs_preview.png"
    status_message = "Generating PNG preview..."
    update_progress(10, "Generating PNG preview...")

    data = fits.getdata(wcs_fits_path)

    # Contrast scaling
    pmin, pmax = np.percentile(data, [20, 80])
    z = ZScaleInterval()
    zmin, zmax = z.get_limits(data)
    vmin = max(zmin, pmin)
    vmax = min(zmax, pmax)
    norm = ImageNormalize(data, vmin=vmin, vmax=vmax, stretch=AsinhStretch())

    plt.figure(figsize=(10, 8))
    plt.imshow(data, cmap="gray", origin="lower", norm=norm)
    plt.title("WCS-Calibrated Image")
    plt.savefig(png_path, dpi=150, bbox_inches="tight")
    plt.close()

    status_message = "Calibration complete!"
    update_progress(100, "Calibration complete!")

    return wcs_fits_path, png_path, ra_center, dec_center, astro_metadata




#======================================
# Functions used for CMD Plot 
#======================================

def mask_apass_region(
    apass_csv,
    ra_center,
    dec_center,
    inner_radius_arcmin=0,
    outer_radius_arcmin=10,
    output_csv="starcluster_calibration.csv"
):
    """
    Create a masked APASS CSV containing only stars within a circular or annular
    region around the cluster center, and visualize the mask using Matplotlib.
    """

    # Load APASS catalog
    df = pd.read_csv(apass_csv)

    # Convert APASS stars to SkyCoord
    star_coords = SkyCoord(df["ra"].values * u.deg,
                           df["dec"].values * u.deg)

    # Cluster center
    center = SkyCoord(ra_center * u.deg, dec_center * u.deg)

    # Angular separation in arcminutes
    sep = star_coords.separation(center).arcminute

    # Mask: keep stars between inner and outer radius
    mask = (sep >= inner_radius_arcmin) & (sep <= outer_radius_arcmin)

    # Save masked CSV
    df_masked = df[mask]
    df_masked.to_csv(output_csv, index=False)

    print(f"Saved masked APASS file: {output_csv}")
    print(f"Stars kept: {len(df_masked)}")

    # ============================
    #       VISUALIZATION
    # ============================

    # Convert RA/Dec to offsets in arcminutes for plotting
    dra = (df["ra"].values - ra_center) * 60  # arcmin
    ddec = (df["dec"].values - dec_center) * 60  # arcmin

    fig, ax = plt.subplots(figsize=(8, 8))

    # Plot all APASS stars
    ax.scatter(dra, ddec, s=10, color="gray", alpha=0.4, label="All APASS stars")

    # Plot masked stars
    ax.scatter(dra[mask], ddec[mask], s=20, color="red", label="Stars inside mask")

    # Draw inner and outer circles
    theta = np.linspace(0, 2*np.pi, 400)
    ax.plot(
        inner_radius_arcmin * np.cos(theta),
        inner_radius_arcmin * np.sin(theta),
        color="blue",
        linestyle="--",
        label="Inner radius"
    )
    ax.plot(
        outer_radius_arcmin * np.cos(theta),
        outer_radius_arcmin * np.sin(theta),
        color="green",
        linestyle="-",
        label="Outer radius"
    )

    # Mark cluster center
    ax.scatter(0, 0, color="yellow", s=80, edgecolor="black", label="Cluster center")

    ax.set_xlabel("ΔRA (arcmin)")
    ax.set_ylabel("ΔDec (arcmin)")
    ax.set_title("APASS Stars and Masked Cluster Region")
    ax.set_aspect("equal")
    ax.legend()
    plt.gca().invert_xaxis()  # RA increases to the left

    return df_masked

def split_apass_cluster_and_calibration(
    apass_csv,
    ra_center,
    dec_center,
    cluster_radius_arcmin,
    outer_radius_arcmin=None,
    cluster_csv="cluster_stars.csv",
    calib_csv="calibration_stars.csv"
):
    """
    Split APASS catalog into:
    - cluster stars (inside cluster_radius_arcmin)
    - calibration stars (outside cluster_radius_arcmin, optionally inside outer_radius_arcmin)
    """

    df = pd.read_csv(apass_csv)

    # Coordinates
    coords = SkyCoord(df["ra"].values * u.deg,
                      df["dec"].values * u.deg)
    center = SkyCoord(ra_center * u.deg, dec_center * u.deg)

    # Separation in arcmin
    sep = coords.separation(center).arcminute

    # Masks
    cluster_mask = sep <= cluster_radius_arcmin

    if outer_radius_arcmin is None:
        calib_mask = sep > cluster_radius_arcmin
    else:
        calib_mask = (sep > cluster_radius_arcmin) & (sep <= outer_radius_arcmin)

    # Save CSVs
    df_cluster = df[cluster_mask]
    df_calib   = df[calib_mask]

    df_cluster.to_csv(cluster_csv, index=False)
    df_calib.to_csv(calib_csv, index=False)

    print(f"Saved cluster stars → {cluster_csv} ({len(df_cluster)} stars)")
    print(f"Saved calibration stars → {calib_csv} ({len(df_calib)} stars)")

    # ============================
    # Visualization
    # ============================
    dra  = (df["ra"].values - ra_center) * 60
    ddec = (df["dec"].values - dec_center) * 60

    fig, ax = plt.subplots(figsize=(8, 8))

    # All stars
    ax.scatter(dra, ddec, s=10, color="gray", alpha=0.3, label="All APASS stars")

    # Cluster stars
    ax.scatter(dra[cluster_mask], ddec[cluster_mask],
               s=25, color="blue", label="Cluster stars")

    # Calibration stars
    ax.scatter(dra[calib_mask], ddec[calib_mask],
               s=25, color="red", label="Calibration stars")

    # Draw cluster radius
    theta = np.linspace(0, 2*np.pi, 400)
    ax.plot(cluster_radius_arcmin * np.cos(theta),
            cluster_radius_arcmin * np.sin(theta),
            color="blue", linestyle="--", label="Cluster radius")

    # Optional outer radius
    if outer_radius_arcmin is not None:
        ax.plot(outer_radius_arcmin * np.cos(theta),
                outer_radius_arcmin * np.sin(theta),
                color="green", linestyle="-", label="Outer radius")

    ax.scatter(0, 0, color="yellow", s=80, edgecolor="black", label="Cluster center")

    ax.set_xlabel("ΔRA (arcmin)")
    ax.set_ylabel("ΔDec (arcmin)")
    ax.set_aspect("equal")
    ax.legend()
    plt.gca().invert_xaxis()
    plt.title("Cluster vs Calibration Star Mask")


    return df_cluster, df_calib

def show_cluster_and_calibration_image(
    image_fits,
    ra_center,
    dec_center,
    cluster_radius_arcmin,
    cluster_csv="cluster_stars.csv",
    calib_csv="calibration_stars.csv"
):
    """
    Display the image with:
    - a circle marking the star cluster radius
    - points for cluster stars
    - points for calibration stars
    """

    # Load image + WCS
    hdu = fits.open(image_fits)[0]
    data = hdu.data
    w = WCS(hdu.header)

    # Load cluster + calibration catalogs
    df_cluster = pd.read_csv(cluster_csv)
    df_calib   = pd.read_csv(calib_csv)

    # Convert RA/Dec to pixel coordinates
    cl_x, cl_y = w.all_world2pix(df_cluster["ra"].values,
                                 df_cluster["dec"].values, 1)
    ca_x, ca_y = w.all_world2pix(df_calib["ra"].values,
                                 df_calib["dec"].values, 1)

    # Cluster center in pixels
    cx, cy = w.all_world2pix(ra_center, dec_center, 1)

    # Convert radius from arcmin to pixels using local scale
    # (approximate: use CD matrix / CDELT)
    # arcsec per pixel from WCS
   # arcsec per pixel from WCS
    # Always returns a clean float array in arcsec/pixel
    # --- UNIVERSAL PIXEL SCALE NORMALIZER ---
    raw_cdelt = w.proj_plane_pixel_scales()

    cdelt_list = []
    for x in raw_cdelt:
        try:
            # Quantity → strip units
            cdelt_list.append(float(x.to(u.deg).value))
        except Exception:
            try:
                # Plain float or ndarray element
                cdelt_list.append(float(x))
            except Exception:
                raise TypeError(f"Unusable pixel scale element: {x}")

    cdelt = np.array(cdelt_list) * 3600.0  # arcsec/pixel
    pix_scale = float(np.mean(cdelt))
    radius_pix = (cluster_radius_arcmin * 60.0) / pix_scale



    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw={"projection": w})
    # Better brightness scaling for star fields
    # Aggressive visibility stretch for very dark FITS images
    data = data.astype(float)

    # Remove background
    med = np.median(data)
    data2 = data - med
    data2[data2 < 0] = 0

    # Scale by a very low percentile (brightens faint stars)
    p = np.percentile(data2[data2 > 0], 0.1)  # 0.1th percentile
    if p <= 0:
        p = np.percentile(data2, 1)

    scaled = data2 / p

    # Asinh stretch with strong boost
    stretched = np.arcsinh(scaled * 5)

    # Normalize to 0–1
    stretched /= stretched.max()

    im = ax.imshow(
        stretched,
        origin="lower",
        cmap="gray",
        vmin=0,
        vmax=1
    )

    

    # Cluster circle
    circ = Circle(
        (cx, cy),
        radius_pix,
        edgecolor=(1, 1, 0, 0.25),   # RGBA: pale yellow, 25% opacity
        facecolor="none",
        linewidth=0.8
    )

    ax.add_patch(circ)

    # Plot cluster stars
    ax.scatter(cl_x, cl_y, s=6, edgecolor="cyan", facecolor="none",
               linewidth=0.2, label="Cluster stars")

    # Plot calibration stars
    ax.scatter(ca_x, ca_y, s=6, color="red", linewidth=0.2, alpha=0.8,
               label="Calibration stars")

    # Mark cluster center
    ax.scatter(cx, cy, s=60, marker="+", color="yellow", linewidth=2,
               label="Cluster center")


    ny, nx = data.shape
    ax.set_xlim(0, nx)
    ax.set_ylim(0, ny)

    ax.set_xlabel("RA")
    ax.set_ylabel("Dec")
    ax.set_title("Cluster Region and Calibration Stars")
    ax.legend(loc="upper right")
    plt.tight_layout()
    output_path = "static/cluster_overlay.png"
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    return output_path

def star_cluster_magnitudes(
    apass_csv,
    green_image,
    red_image,
    ra_center,
    dec_center,
    inner_radius_arcmin,
    outer_radius_arcmin,
    output_calib_csv="cluster_calibrated_magnitudes.csv"
):
    
    global status_message

    calib_cat = pd.read_csv("calibration_stars.csv")
    ap_ra  = calib_cat["ra"].values
    ap_dec = calib_cat["dec"].values
    g_std  = calib_cat["mag_g"].values
    r_std  = calib_cat["mag_r"].values

    w_g = WCS(green_image)
    w_r = WCS(red_image)

    ap_x_g, ap_y_g = w_g.all_world2pix(ap_ra, ap_dec, 1)
    ap_x_r, ap_y_r = w_r.all_world2pix(ap_ra, ap_dec, 1)

    green_flux = flux(ap_x_g, ap_y_g, 5, green_image)
    red_flux   = flux(ap_x_r, ap_y_r, 5, red_image)

    good = (
        (green_flux > 0) &
        np.isfinite(green_flux) &
        (red_flux > 0) &
        np.isfinite(red_flux)
    )

    green_flux = green_flux[good]
    red_flux   = red_flux[good]
    g_std      = g_std[good]
    r_std      = r_std[good]

    inst_g = -2.5 * np.log10(green_flux)
    inst_r = -2.5 * np.log10(red_flux)

    inst_g_r = inst_g - inst_r
    st_g_r   = g_std - r_std
    g_offset = g_std - inst_g

    Tgr, Cgr, err_color = lsrl(inst_g_r, st_g_r)
    Tg,  Cg,  err_g     = lsrl(st_g_r, g_offset)

    sources = detect_stars(green_image, threshold=3.0)
    det_x = np.array(sources["xcentroid"])
    det_y = np.array(sources["ycentroid"])

    det_ra, det_dec = w_g.all_pix2world(det_x, det_y, 1)
    det_coords = SkyCoord(det_ra*u.deg, det_dec*u.deg)
    center = SkyCoord(ra_center*u.deg, dec_center*u.deg)
    sep = det_coords.separation(center).arcminute

    cluster_mask = sep <= inner_radius_arcmin

    cl_x = det_x[cluster_mask]
    cl_y = det_y[cluster_mask]
    cl_ra = det_ra[cluster_mask]
    cl_dec = det_dec[cluster_mask]

    cl_flux_g = flux(cl_x, cl_y, 5, green_image)
    cl_x_r, cl_y_r = w_r.all_world2pix(cl_ra, cl_dec, 1)
    cl_flux_r = flux(cl_x_r, cl_y_r, 5, red_image)

    good_cl = (
        (cl_flux_g > 0) &
        np.isfinite(cl_flux_g) &
        (cl_flux_r > 0) &
        np.isfinite(cl_flux_r)
    )

    cl_flux_g = cl_flux_g[good_cl]
    cl_flux_r = cl_flux_r[good_cl]
    cl_ra     = cl_ra[good_cl]
    cl_dec    = cl_dec[good_cl]

    cl_inst_g = -2.5 * np.log10(cl_flux_g)
    cl_inst_r = -2.5 * np.log10(cl_flux_r)
    cl_inst_g_r = cl_inst_g - cl_inst_r

    cl_std_g_r = Tgr * cl_inst_g_r + Cgr
    cl_std_g   = Tg * cl_std_g_r + Cg + cl_inst_g
    cl_std_r   = cl_std_g - cl_std_g_r

    # ============================================================
    # CMD plot
    # ============================================================
    plt.figure(figsize=(7,6))
    plt.scatter(cl_std_g_r, cl_std_g, s=12, alpha=0.8)
    plt.gca().invert_yaxis()
    plt.xlabel("Standard (g - r)")
    plt.ylabel("Standard g")
    plt.title("Cluster CMD (Calibrated)")
    cmd_path = "static/cmd_plot.png"
    plt.savefig(cmd_path, dpi=150, bbox_inches="tight")
    plt.close()

    # ============================================================
    # Color calibration plot
    # ============================================================
    plt.figure(figsize=(7,5))
    plt.scatter(inst_g_r, st_g_r, s=15, alpha=0.7)
    plt.plot(inst_g_r, Tgr*inst_g_r + Cgr, color="red")
    plt.text(
        0.05, 0.95,
        f"Tgr={Tgr:.4f}\nCgr={Cgr:.4f}",
        transform=plt.gca().transAxes,
        fontsize=12,
        verticalalignment="top",
        bbox=dict(facecolor="black", alpha=0.3, edgecolor="white")
    )
    color_path = "static/color_calibration.png"
    plt.savefig(color_path, dpi=150, bbox_inches="tight")
    plt.close()

    # ============================================================
    # Green offset plot
    # ============================================================
    plt.figure(figsize=(7,5))
    plt.scatter(st_g_r, g_offset, s=15, alpha=0.7)
    plt.plot(st_g_r, Tg*st_g_r + Cg, color="red")
    plt.text(
        0.05, 0.95,
        f"Tg={Tg:.4f}\nCg={Cg:.4f}",
        transform=plt.gca().transAxes,
        fontsize=12,
        verticalalignment="top",
        bbox=dict(facecolor="black", alpha=0.3, edgecolor="white")
    )
    offset_path = "static/green_offset.png"
    plt.savefig(offset_path, dpi=150, bbox_inches="tight")
    plt.close()



    status_message = "Done!"

    return cmd_path, color_path, offset_path, Tgr, Cgr, Tg, Cg



#============================
# Create and running app using flask
#============================

app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def home():
    user_text = None

    if request.method == "POST":
        user_text = request.form.get("user_input")

    return render_template("index.html", user_text=user_text)

@app.route("/object_calibration", methods=["GET", "POST"])
def object_calibration():
    global last_g, last_r, last_err_g, last_err_r, last_ra, last_dec
    global last_ra, last_dec, last_ra_hms, last_dec_dms
    global error_message
    global status_message
    global progress_value, progress_message
                
    user_text = None
    g_text = None
    r_text = None
    g_file = None
    r_file = None
    g_path = None 
    r_path = None

    standard_g_target = None 
    standard_r_target = None 
    error_g = None 
    error_r = None
    Tgr = None
    Cgr = None
    Tg = None
    Cg = None


    ra_decimal = None
    dec_decimal = None
    ra_hms = None
    dec_dms = None

    color_term_path = None
    green_offset_path = None
    last_target_name = None
    last_target_cutout = None


    red_wcs_path = None
    green_wcs_path = None
    color_term_path = None
    green_offset_path = None


    ra_deg = None
    dec_deg = None
    ra_center = None
    dec_center = None


    upload_folder = "uploads" 
    os.makedirs(upload_folder, exist_ok=True)

    

    if request.method == "POST":

        # Reset progress bar 
        global progress_value, progress_message
        progress_value = 0 
        progress_message = "Starting..."


        # Single input case
        user_text = request.form.get("calibration_input")

        
        calibration_mode = request.form.get("calibration_mode")
        calibration_count = request.form.get("calibration_count")



        # Both-input case
        g_text = request.form.get("g_link_both")
        r_text = request.form.get("r_link_both")

        g_file = request.files.get("g_file")
        r_file = request.files.get("r_file")

        ra_decimal = request.form.get("ra_decimal")
        dec_decimal = request.form.get("dec_decimal")
        ra_hms = request.form.get("ra_hms")
        dec_dms = request.form.get("dec_dms")

        # Option A: decimal degrees
        if ra_decimal and dec_decimal:
            try:
                ra_deg = float(ra_decimal)
                dec_deg = float(dec_decimal)

                last_ra = ra_deg
                last_dec = dec_deg

                # Convert to sexagesimal for AAVSO
                coord = SkyCoord(ra_deg * u.deg, dec_deg * u.deg)
                last_ra_hms = coord.ra.to_string(unit=u.hour, sep=":", precision=2)
                last_dec_dms = coord.dec.to_string(unit=u.deg, sep=":", precision=2, alwayssign=True)

            except ValueError:
                pass

        # Option B: HMS/DMS
        elif ra_hms and dec_dms:
            try:
                coord = SkyCoord(ra_hms, dec_dms, unit=(u.hourangle, u.deg))
                ra_deg = coord.ra.deg
                dec_deg = coord.dec.deg
            except Exception:
                pass

        # If neither option was provided
        if ra_deg is None or dec_deg is None:

            return render_template(
                "object_calibration.html",
                error_message="Please enter RA/Dec in either decimal or HMS/DMS format."


            )
        

 

    if g_file and g_file.filename.strip():
        g_path = os.path.join(upload_folder, "uploaded_green.fits")
        g_file.save(g_path)

    if r_file and r_file.filename.strip():
        r_path = os.path.join(upload_folder, "uploaded_red.fits")
        r_file.save(r_path)


    # If user uploaded files, use those
    if g_path and r_path:
        full_calibration_with_subid(g_path, "wcs_green_solution.fits", os.environ.get("GREEN_SUBID")) 
        num_rows = full_calibration_with_subid(r_path, "wcs_red_solution.fits", os.environ.get("RED_SUBID")) 
        #full_calibration(g_path, "wcs_green_solution.fits")
        #num_rows =full_calibration(r_path, "wcs_red_solution.fits")

        # ============================
        # CHECK IF TARGET IS IN IMAGE
        # ============================

        # Only run this check if WCS files already exist
        if os.path.exists("wcs_green_solution.fits"):
            w_g = WCS("wcs_green_solution.fits")

            with fits.open("wcs_green_solution.fits") as hdul:
                ny, nx = hdul[0].data.shape

            # Convert RA/Dec to pixel coordinates
            tx, ty = w_g.all_world2pix(ra_deg, dec_deg, 1)


            if calibration_mode == "custom":
                try:
                    n = int(calibration_count)
                    n = max(1, n)  # at least 1
                    num_rows = min(n, num_rows)  # cannot exceed available stars
                except:
                    num_rows = num_rows
            else:
                num_rows = num_rows


            # Out-of-bounds check
            if tx < 0 or tx >= nx or ty < 0 or ty >= ny:

                    status_message = "❌ The target RA/Dec is outside the image boundaries."

                    error_message = "❌ The target RA/Dec (Star coordinates) are outside the image boundaries. Please retry with different coordinates or with a different image."

                    return render_template(
                        "object_calibration.html",
                        error_message=error_message, 
                        status_message=status_message
                    )


        standard_g_target, standard_r_target, error_g, error_r, Tgr, Cgr, Tg, Cg, color_term_path, green_offset_path, red_wcs_path, green_wcs_path, last_target_name, last_target_cutout, ra_center, dec_center  = magnitudes(
            "apass_subset.csv",
            "wcs_green_solution.fits",
            "wcs_red_solution.fits",
            num_rows,
            ra_deg,
            dec_deg
        )

    # If user typed paths instead, use those
    elif g_text and r_text:
        
        full_calibration_with_subid(g_text, "wcs_green_solution.fits", os.environ.get("GREEN_SUBID"))  # GREEN_SUBID_JUL15  GREEN_SUBID_JUL16 GREEN_SUBID_NGC
        num_rows = full_calibration_with_subid(r_text, "wcs_red_solution.fits", os.environ.get("RED_SUBID")) # RED_SUBID_JUL15 RED_SUBID_JUL16 RED_SUBID_NGC
        
        
        
        #full_calibration(g_text, "wcs_green_solution.fits")
        #num_rows = full_calibration(r_text, "wcs_red_solution.fits")

        # ============================
        # CHECK IF TARGET IS IN IMAGE
        # ============================

        # Only run this check if WCS files already exist
        if os.path.exists("wcs_green_solution.fits"):
            w_g = WCS("wcs_green_solution.fits")

            with fits.open("wcs_green_solution.fits") as hdul:
                ny, nx = hdul[0].data.shape

            # Convert RA/Dec to pixel coordinates
            tx, ty = w_g.all_world2pix(ra_deg, dec_deg, 1)

            # Out-of-bounds check
            if tx < 0 or tx >= nx or ty < 0 or ty >= ny:

                    status_message = "❌ The target RA/Dec is outside the image boundaries."

                    error_message = "❌ The target RA/Dec is outside the image boundaries."

                    return render_template(
                        "object_calibration.html",
                        error_message=error_message, 
                        status_message=status_message
                    )

            
        if calibration_mode == "custom":
            try:
                n = int(calibration_count)
                n = max(1, n)  # at least 1
                num_rows = min(n, num_rows)  # cannot exceed available stars
            except:
                num_rows = num_rows
        else:
            num_rows = num_rows


        standard_g_target, standard_r_target, error_g, error_r, Tgr, Cgr, Tg, Cg, color_term_path, green_offset_path, red_wcs_path, green_wcs_path, last_target_name, last_target_cutout, ra_center, dec_center = magnitudes(
            "apass_subset.csv",
            "wcs_green_solution.fits",
            "wcs_red_solution.fits",
            num_rows,
            ra_deg,
            dec_deg
        )

    

    last_g = standard_g_target
    last_r = standard_r_target
    last_err_g = error_g
    last_err_r = error_r
    last_ra = ra_deg
    last_dec = dec_deg

    return render_template(
        "object_calibration.html", 
        standard_g_target=standard_g_target,
        standard_r_target=standard_r_target,
        error_g=error_g,
        error_r=error_r,
        Tgr=Tgr,
        Cgr=Cgr,
        Tg=Tg,
        Cg=Cg,
        color_term_path=color_term_path,
        green_offset_path=green_offset_path, 
        red_wcs_path=red_wcs_path, 
        green_wcs_path=green_wcs_path, 
        ra_deg=ra_deg, 
        dec_deg=dec_deg, 
        error_message=error_message, 
        last_target_name=last_target_name, 
        last_target_cutout=last_target_cutout,
        g_path_name=g_file,
        r_path_name=r_file,
        g_text_name=g_text,
        r_text_name=r_text,
        ra_center=ra_center,
        dec_center=dec_center

    )


#============================
# FOR STAR CLUSTER 
#============================
@app.route("/star_cluster_calibration", methods=["GET", "POST"])
def star_cluster_calibration():
    user_text = None
    # -----------------------------
    # Initialize variables
    # -----------------------------
    g_text = None
    r_text = None
    g_file = None
    r_file = None
    g_path = None
    r_path = None

    ra_deg = None
    dec_deg = None
    cluster_radius_arcmin = None



    upload_folder = "uploads"
    os.makedirs(upload_folder, exist_ok=True)

    # -----------------------------
    # Handle POST
    # -----------------------------
    if request.method == "POST":

        # -----------------------------
        # 1. Get RA/Dec inputs
        # -----------------------------
        ra_decimal = request.form.get("ra_decimal")
        dec_decimal = request.form.get("dec_decimal")
        ra_hms = request.form.get("ra_hms")
        dec_dms = request.form.get("dec_dms")

        # Decimal degrees
        if ra_decimal and dec_decimal:
            try:
                ra_deg = float(ra_decimal)
                dec_deg = float(dec_decimal)
            except ValueError:
                return render_template(
                    "star_cluster_calibration.html",
                    error_message="Invalid decimal RA/Dec format."
                )

        # HMS/DMS
        elif ra_hms and dec_dms:
            try:
                coord = SkyCoord(ra_hms, dec_dms, unit=(u.hourangle, u.deg))
                ra_deg = coord.ra.deg
                dec_deg = coord.dec.deg
            except Exception:
                return render_template(
                    "star_cluster_calibration.html",
                    error_message="Invalid HMS/DMS RA/Dec format."
                )

        else:
            return render_template(
                "star_cluster_calibration.html",
                error_message="Please enter RA/Dec in decimal or HMS/DMS format."
            )

        # -----------------------------
        # 2. Cluster radius (arcmin)
        # -----------------------------
        cluster_radius_arcmin = request.form.get("cluster_radius_arcmin")
        if not cluster_radius_arcmin:

            status_message = "Please enter a cluster radius in arcminutes"
            return render_template(
                "star_cluster_calibration.html",
                error_message="Please enter a cluster radius in arcminutes.", 
                status_message=status_message
            )

        try:
            cluster_radius_arcmin = float(cluster_radius_arcmin)
        except ValueError:
            return render_template(
                "star_cluster_calibration.html",
                error_message="Cluster radius must be a number."
            )

        outer_radius_arcmin = cluster_radius_arcmin * 1.5

        # -----------------------------
        # 3. Get image inputs
        # -----------------------------
        g_text = request.form.get("g_link")
        r_text = request.form.get("r_link")

        g_file = request.files.get("g_file")
        r_file = request.files.get("r_file")

        # Uploaded files
        if g_file and g_file.filename.strip():
            g_path = os.path.join(upload_folder, g_file.filename)
            g_file.save(g_path)

        if r_file and r_file.filename.strip():
            r_path = os.path.join(upload_folder, r_file.filename)
            r_file.save(r_path)

        # -----------------------------
        # 4. Determine image sources
        # -----------------------------
        if g_path and r_path:
            green_image = g_path
            red_image = r_path

            #full_calibration(green_image, "wcs_green_solution.fits")
            #full_calibration(red_image, "wcs_red_solution.fits")

            full_calibration_with_subid(green_image, "wcs_green_solution.fits", os.environ.get("GREEN_SUBID_SC"))
            full_calibration_with_subid(red_image, "wcs_red_solution.fits", os.environ.get("RED_SUBID_SC"))

        elif g_text and r_text:
            green_image = g_text
            red_image = r_text

            full_calibration(green_image, "wcs_green_solution.fits")
            full_calibration(red_image, "wcs_red_solution.fits")
            #full_calibration_with_subid(green_image, "wcs_green_solution.fits", os.environ.get("GREEN_SUBID_SC"))
            #full_calibration_with_subid(red_image, "wcs_red_solution.fits", os.environ.get("RED_SUBID_SC"))

            # ============================
            # CHECK IF TARGET IS IN IMAGE
            # ============================

            # Only run this check if WCS files already exist
            if os.path.exists("wcs_green_solution.fits"):
                w_g = WCS("wcs_green_solution.fits")

                with fits.open("wcs_green_solution.fits") as hdul:
                    ny, nx = hdul[0].data.shape

                # Convert RA/Dec to pixel coordinates
                tx, ty = w_g.all_world2pix(ra_deg, dec_deg, 1)

                # Out-of-bounds check
                if tx < 0 or tx >= nx or ty < 0 or ty >= ny:

                    status_message = "❌ The target RA/Dec is outside the image boundaries."

                    error_message = "❌ The target RA/Dec is outside the image boundaries."

                    return render_template(
                        "object_calibration.html",
                        error_message=error_message, 
                        status_message=status_message
                    )


        else:
            return render_template(
                "star_cluster_calibration.html",
                error_message="Please upload or link BOTH Sloan g and r images."
            )

        # -----------------------------
        # 5. Split APASS into cluster + calibration
        # -----------------------------
        split_apass_cluster_and_calibration(
            "apass_subset.csv",
            ra_center=ra_deg,
            dec_center=dec_deg,
            cluster_radius_arcmin=cluster_radius_arcmin,
            outer_radius_arcmin=outer_radius_arcmin
        )

        # -----------------------------
        # 6. Show cluster + calibration overlay
        # -----------------------------
        overlay_path = show_cluster_and_calibration_image(
            "wcs_green_solution.fits",
            ra_center=ra_deg,
            dec_center=dec_deg,
            cluster_radius_arcmin=cluster_radius_arcmin,
            cluster_csv="cluster_stars.csv",
            calib_csv="calibration_stars.csv"
        )


        # -----------------------------
        # 7. Run final star cluster photometry
        # -----------------------------
        cmd_path, color_path, offset_path, Tgr, Cgr, Tg, Cg = star_cluster_magnitudes(
            "apass_subset.csv",
            "wcs_green_solution.fits",
            "wcs_red_solution.fits",
            ra_center=ra_deg,
            dec_center=dec_deg,
            inner_radius_arcmin=cluster_radius_arcmin,
            outer_radius_arcmin=outer_radius_arcmin
        )

        # -----------------------------
        # 8. Render results
        # -----------------------------
        return render_template(
            "star_cluster_calibration.html",
            cmd_path=cmd_path,
            color_path=color_path,
            offset_path=offset_path,
            overlay_path=overlay_path,
            Tgr=Tgr, 
            Cgr=Cgr, 
            Tg=Tg, 
            Cg=Cg
        )


    # -----------------------------
    # GET request
    # -----------------------------
    return render_template("star_cluster_calibration.html")


#============================
# FOR CALIBRATING IMAGES
#============================

@app.route("/calibrate_image", methods=["GET", "POST"])
def calibrate_image_page():
    if request.method == "POST":
        global progress_value, progress_message
        progress_value = 0 
        progress_message = "Starting..."

        
        file = request.files.get("fits_file")
        if not file:
            return render_template("calibrate_image.html", error="No file uploaded.")

        # ⭐ Preserve original filename for display
        original_name = file.filename

        # ⭐ Always overwrite the same uploaded file
        upload_path = "uploads/uploaded_image.fits"
        file.save(upload_path)

        # ⭐ Run calibration (always overwrites the same output files)
        wcs_fits, png_path, ra_center, dec_center, astro_meta = calibrate_image_only(upload_path)

        # ⭐ Convert center coordinates to sexagesimal
        coord = SkyCoord(ra_center * u.deg, dec_center * u.deg)
        ra_hms = coord.ra.to_string(unit=u.hour, sep=":", precision=2)
        dec_dms = coord.dec.to_string(unit=u.deg, sep=":", precision=2, alwayssign=True)

        return render_template(
            "calibrate_image.html",
            original_name=original_name,      # ⭐ original uploaded filename
            uploaded_path=upload_path,        # ⭐ fixed internal filename
            wcs_fits=wcs_fits,                # ⭐ fixed output filename
            wcs_png=png_path,                 # ⭐ fixed output PNG
            ra_center=ra_center,
            dec_center=dec_center,
            ra_hms=ra_hms,
            dec_dms=dec_dms,
            astro_meta=astro_meta
        )

    return render_template("calibrate_image.html")


@app.route("/status")
def status():
    global status_message
    return status_message

@app.route("/progress_status")
def progress_status():
    global progress_value, progress_message
    return jsonify({
        "message": progress_message,
        "progress": progress_value
    })

# ============================
# For transferring the data to submission page
# ============================

@app.route("/aavso_instructions")
def aavso_instructions():

    # Global storage for last calibration results
    

    return render_template(
        "submit_instructions.html",
        g_mag=last_g,
        r_mag=last_r,
        err_g=last_err_g,
        err_r=last_err_r,
        ra=last_ra,
        dec=last_dec,
        jd=last_jd, 
        ra_hms=last_ra_hms,
        dec_dms=last_dec_dms, 
        last_target_name=last_target_name



    )

# ============================
# Julian Date Conversion Route
# ============================

@app.route("/calculate_jd", methods=["POST"])
def calculate_jd():
    global last_jd

    date_str = request.form.get("obs_date")
    time_str = request.form.get("obs_time")

    if not date_str or not time_str:
        last_jd = None
    else:
        try:
            t = Time(f"{date_str} {time_str}", format="iso", scale="utc")
            last_jd = t.jd
            print("JD calculated:", last_jd)
        except Exception as e:
            print("JD conversion error:", e)
            last_jd = None

    return redirect("/aavso_instructions")


@app.route("/convert_radec", methods=["POST"])
def convert_radec():
    global last_ra_hms, last_dec_dms

    ra_deg = request.form.get("ra_deg")
    dec_deg = request.form.get("dec_deg")

    try:
        coord = SkyCoord(ra=float(ra_deg)*u.deg, dec=float(dec_deg)*u.deg)
        last_ra_hms = coord.ra.to_string(unit=u.hour, sep=":", precision=2)
        last_dec_dms = coord.dec.to_string(unit=u.deg, sep=":", precision=2, alwayssign=True)
    except Exception:
        last_ra_hms = None
        last_dec_dms = None

    return redirect("/aavso_instructions")


if __name__ == "__main__":
    app.run()


