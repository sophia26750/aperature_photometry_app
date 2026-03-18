from flask import Flask, render_template, request

import requests
import json
import os

import time
import random
import math

from astropy.io import fits 
from astropy.stats import sigma_clipped_stats
from astropy.io import fits
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord
from astropy.table import Table
from astropy.io import ascii
import astropy.units as u

from photutils.aperture import CircularAperture, CircularAnnulus, aperture_photometry
from astropy.stats import SigmaClip
from photutils.background import Background2D, MedianBackground


from matplotlib.patches import Circle
import matplotlib.pyplot as plt 
import matplotlib.lines as mlines
import numpy as np
import pyvo
import pandas as pd

from photutils.detection import DAOStarFinder

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
            print("Job found:", jobs[0])

            
            print(f"Job found: {jobs[0]}")


            return jobs[0]
        print("Waiting for job...", i)        
        print(f"Waiting to achieve Job ID... {i}")

        time.sleep(2)
    raise TimeoutError("Job did not appear in time.")


def wait_for_calibration(job_id, timeout=180):
    url = f"http://nova.astrometry.net/api/jobs/{job_id}/calibration/"
    
    for _ in range(timeout):
        r = safe_get_json(url)
        if r is None:
            print("Error: Could not retrieve calibration status.")
            return None


        # Calibration is ready
        if "ra" in r:
            return r

        time.sleep(2)

    return None



def apply_calibration_to_fits(input_fits, output_fits, job_id):
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
    return wcs


def query_apass_to_csv(ra_center, dec_center, radius_deg, output_csv="apass_subset.csv"):

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
    print(f"Saved csv file as {output_csv}...")
    num_rows = len(df)
    print(f"Done!")

    return num_rows - 2

def full_calibration_with_subid(image, wcs_image_name, subid_key):

    # 1. Login to Astrometry.net
    api_key = os.environ.get("ASTRO_LOGIN")
    session_key = login_to_astrometry(api_key)
    print("Session:", session_key)
    print(f"Session key achieved...please wait for submission ID")


    # 2. Upload image if no subid_key provided
    if subid_key is None:
        subid_key = upload_fits_file(image, session_key)
        print("Submission ID:", subid_key)

    else:
        subid_key = subid_key
        print("Submission ID:", subid_key)


    job_id = wait_for_job(subid_key)
    astro_results = wait_for_calibration(job_id)

    w = apply_calibration_to_fits(
        image,
        wcs_image_name,
        job_id=job_id
    )


    # 6. Query APASS around the solved coordinates
    ra = astro_results["ra"]
    dec = astro_results["dec"]
    radius = round(astro_results["radius"] * 0.6, 1)

    num_rows = query_apass_to_csv(ra, dec, radius, "apass_subset.csv")
    return num_rows


def full_calibration(image, wcs_image_name, subid_key=None):

    # 1. Login to Astrometry.net
    api_key = os.environ.get("ASTRO_LOGIN")
    session_key = login_to_astrometry(api_key)
    print("Session:", session_key)
    print(f"Session key achieved...please wait for submission ID")


    # 2. Upload image if no subid_key provided
    if subid_key is None:
        subid_key = upload_fits_file(image, session_key)
        print("Submission ID:", subid_key)
        if subid_key is None:
            print("ERROR: Astrometry.net upload failed — no submission ID returned.")
            return None


    else:
        subid_key = subid_key
        print("Submission ID:", subid_key)


    job_id = wait_for_job(subid_key)
    astro_results = wait_for_calibration(job_id)

    w = apply_calibration_to_fits(
        image,
        wcs_image_name,
        job_id=job_id
    )


    # 6. Query APASS around the solved coordinates
    ra = astro_results["ra"]
    dec = astro_results["dec"]
    radius = round(astro_results["radius"] * 0.7, 1)

    num_rows = query_apass_to_csv(ra, dec, radius, "apass_subset.csv")
    return num_rows

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

def show_flux_cutouts(image, x, y, radius, n_show=12):
    """
    Visualize the stars used for flux measurement.
    Shows each star with:
        - the circular aperture
        - the background annulus
    """
    data = fits.getdata(image)
    ny, nx = data.shape

    # choose a subset if too many
    N = len(x)
    n_show = min(n_show, N)
    idx = np.random.choice(N, n_show, replace=False)

    fig, axes = plt.subplots(3, 4, figsize=(12, 9))
    axes = axes.flatten()

    for k, ax in enumerate(axes[:n_show]):
        i = idx[k]
        cx, cy = x[i], y[i]

        # cutout size = 8×radius
        half = int(radius * 4)
        x1, x2 = int(cx - half), int(cx + half)
        y1, y2 = int(cy - half), int(cy + half)

        # bounds check
        x1 = max(x1, 0); y1 = max(y1, 0)
        x2 = min(x2, nx); y2 = min(y2, ny)

        cut = data[y1:y2, x1:x2]

        ax.imshow(cut, cmap="gray", origin="lower",
                  vmin=np.percentile(cut, 5),
                  vmax=np.percentile(cut, 99))

        # aperture circle
        circ = plt.Circle((cx - x1, cy - y1), radius,
                          edgecolor='yellow', facecolor='none', lw=1.5)
        ax.add_patch(circ)

        # annulus
        ann = plt.Circle((cx - x1, cy - y1), radius+3,
                         edgecolor='cyan', facecolor='none', lw=1)
        ax.add_patch(ann)
        ann2 = plt.Circle((cx - x1, cy - y1), radius+6,
                          edgecolor='cyan', facecolor='none', lw=1)
        ax.add_patch(ann2)

        ax.set_title(f"Star {i}")
        ax.set_xticks([]); ax.set_yticks([])

    plt.tight_layout()
    plt.show()


def show_three_labeled_stars(image, x, y, radius=5, n_show=3):
    """
    Show n_show stars in SEPARATE figures.
    Each figure includes:
        - yellow aperture
        - cyan annulus
        - red pixel labels (4 decimals)
    """

    data = fits.getdata(image)
    ny, nx = data.shape

    N = len(x)
    n_show = min(n_show, N)

    # Randomly choose stars
    idx = np.random.choice(N, n_show, replace=False)

    for count, i in enumerate(idx):
        cx, cy = x[i], y[i]

        # Zoom window
        half = int(radius * 4)
        x1, x2 = int(cx - half), int(cx + half)
        y1, y2 = int(cy - half), int(cy + half)

        x1 = max(x1, 0); y1 = max(y1, 0)
        x2 = min(x2, nx); y2 = min(y2, ny)

        cut = data[y1:y2, x1:x2]

        plt.figure(figsize=(6, 6))
        plt.imshow(cut, cmap="gray", origin="lower",
                   vmin=np.percentile(cut, 5),
                   vmax=np.percentile(cut, 99))

        # --- Aperture ring (yellow) ---
        circ = plt.Circle((cx - x1, cy - y1), radius,
                          edgecolor='yellow', facecolor='none', lw=1.5)
        plt.gca().add_patch(circ)

        # --- Annulus rings (cyan) ---
        ann1 = plt.Circle((cx - x1, cy - y1), radius+3,
                          edgecolor='cyan', facecolor='none', lw=1)
        ann2 = plt.Circle((cx - x1, cy - y1), radius+6,
                          edgecolor='cyan', facecolor='none', lw=1)
        plt.gca().add_patch(ann1)
        plt.gca().add_patch(ann2)

        # --- Pixel labels (red, tiny) ---
        for j in range(cut.shape[0]):
            for ii in range(cut.shape[1]):
                plt.text(ii, j, f"{cut[j, ii]:.4f}",
                         color="red",
                         fontsize=3,
                         ha="center", va="center")

        plt.title(f"Star {i}")
        plt.xticks([]); plt.yticks([])
        plt.tight_layout()
        plt.show()


def lsrl(x, y):
    x = np.array(x)
    y = np.array(y)

    if np.std(x) < 1e-3 or np.std(y) < 1e-3:
        raise ValueError("Not enough color variation for calibration.")

    m, b = np.polyfit(x, y, 1)
    sd = np.std(y - (m*x + b))
    return m, b, sd

def show_single_cutout(image, x, y, radius):
    data = fits.getdata(image)
    ny, nx = data.shape

    half = int(radius * 4)
    x1, x2 = int(x - half), int(x + half)
    y1, y2 = int(y - half), int(y + half)

    x1 = max(x1, 0); y1 = max(y1, 0)
    x2 = min(x2, nx); y2 = min(y2, ny)

    cut = data[y1:y2, x1:x2]

    plt.figure(figsize=(5,5))
    plt.imshow(cut, cmap="gray", origin="lower",
               vmin=np.percentile(cut, 5),
               vmax=np.percentile(cut, 99))

    circ = plt.Circle((x - x1, y - y1), radius,
                      edgecolor='yellow', facecolor='none', lw=2)
    plt.gca().add_patch(circ)

    ann1 = plt.Circle((x - x1, y - y1), radius+3,
                      edgecolor='cyan', facecolor='none', lw=1)
    ann2 = plt.Circle((x - x1, y - y1), radius+6,
                      edgecolor='cyan', facecolor='none', lw=1)
    plt.gca().add_patch(ann1)
    plt.gca().add_patch(ann2)

    plt.title("Target Cutout")
    plt.xticks([]); plt.yticks([])
    plt.tight_layout()
    plt.show()


def magnitudes(csv_file, green_image, red_image, n, RA, DEC):
    data = ascii.read(csv_file, format='csv')
    hdul_g = fits.open(green_image)
    hdul_r = fits.open(red_image)
    wcs_g_h = WCS(hdul_g[0].header)
    wcs_r_h = WCS(hdul_r[0].header)

    fits_file_path = green_image
    with fits.open(fits_file_path) as hdul:
        image_data = hdul[0].data

    w_g = WCS("wcs_green_solution.fits") 
    w_r = WCS("wcs_red_solution.fits")


    col_length = len(data['ra'])

    # Ensure n never exceeds available stars
    n = min(n, col_length)

    calibration_num = random.sample(range(col_length), n)

    calibration_num = sorted(calibration_num)
    print (f"Calibration numbers: {calibration_num}")

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

    # === Visualize RED WCS solution ===
    hdul_r = fits.open(red_image)
    image_data_r = hdul_r[0].data
    vmin_r, vmax_r = np.percentile(image_data_r, [5, 99])

    fig = plt.figure(figsize=(10,8))
    ax = plt.subplot(projection=w_r)
    ax.imshow(image_data_r, cmap="gray", origin="lower", vmin=vmin_r, vmax=vmax_r)

    # Plot calibration stars
    ax.scatter(x_pixel_r, y_pixel_r, s=80, edgecolor='cyan', facecolor='none', linewidth=1.5, label="Calibration stars")

    # --- NEW: plot target object ---
    target_r_x, target_r_y = w_r.all_world2pix(RA, DEC, 1)
    target_circle_r = plt.Circle((target_r_x, target_r_y), 12,
                                edgecolor='yellow', facecolor='none',
                                linewidth=2.5)
    ax.add_patch(target_circle_r)
    ax.text(target_r_x + 10, target_r_y + 10, "Target", color='yellow')

    plt.title("RED WCS Check: APASS stars + Target")
    plt.xlabel("RA")
    plt.ylabel("Dec")
    plt.legend()
    plt.show()



            
    vmin, vmax = np.percentile(image_data, [5, 99])
    fig = plt.figure(figsize=(10,8))
    ax = plt.subplot(projection=w_g)
    ax.imshow(image_data, cmap="gray", origin="lower", vmin=vmin, vmax=vmax)

    # Plot calibration stars
    ax.scatter(x_pixel_g, y_pixel_g, s=50, edgecolor='red', facecolor='none', label="Calibration stars")

    # --- NEW: plot target object ---
    target_g_x, target_g_y = w_g.all_world2pix(RA, DEC, 1)
    target_circle_g = plt.Circle((target_g_x, target_g_y), 12,
                                edgecolor='yellow', facecolor='none',
                                linewidth=2.5)
    ax.add_patch(target_circle_g)
    ax.text(target_g_x + 10, target_g_y + 10, "Target", color='yellow')

    plt.title("GREEN WCS Check: APASS stars + Target")
    plt.xlabel("RA")
    plt.ylabel("Dec")
    plt.legend()
    plt.show()




    # === 1. Remove APASS stars outside the image ===
    ny, nx = image_data.shape

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
    print("APASS stars inside image:", len(x_pixel_g))


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



        
    green_flux = flux(x_green, y_green, 7, green_image)
    show_flux_cutouts(green_image, x_green, y_green, radius=5)

    red_flux = flux(x_red, y_red, 7, red_image)
    show_flux_cutouts(red_image, x_red, y_red, radius=5)



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



    valid_flux_g = np.array(valid_flux_g)
    valid_flux_r = np.array(valid_flux_r)

    inst_g = -2.5 * np.log10(valid_flux_g)
    inst_r = -2.5 * np.log10(valid_flux_r)


    #X1
    inst_g_r = inst_g - inst_r
    
    # Y1 & X2
    st_g_r = g - r
    
    # Y2
    g_offset = g - inst_g


    # ============================================
    # LIMIT TO n CALIBRATION STARS *AFTER MATCHING*
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

    # Compute target pixel coordinates
    target_pixel_g_x, target_pixel_g_y = w_g.all_world2pix(RA, DEC, 1)
    target_pixel_r_x, target_pixel_r_y = w_r.all_world2pix(RA, DEC, 1)

    # --- SHOW TARGET CUTOUTS ---
    show_single_cutout(
        green_image,
        target_pixel_g_x,
        target_pixel_g_y,
        radius=5
    )

    show_single_cutout(
        red_image,
        target_pixel_r_x,
        target_pixel_r_y,
        radius=5
    )




    target_flux_g = target_flux(target_pixel_g_x, target_pixel_g_y, 8, green_image)
    target_flux_r = target_flux(target_pixel_r_x, target_pixel_r_y, 8, red_image)

    target_g_inst_mag = -2.5 * np.log10(target_flux_g)
    target_r_inst_mag = -2.5 * np.log10(target_flux_r)

    print("std(inst_g_r) =", np.std(inst_g_r))
    print("std(st_g_r)   =", np.std(st_g_r))


    m1_b1 = lsrl(inst_g_r, st_g_r)
    m2_b2 = lsrl(st_g_r, g_offset)

    new_std = m1_b1[0]*(inst_g_r) + m1_b1[1]
    new_std_inst = m2_b2[0]*new_std + m2_b2[1] 
    label_text = f'Tgr = {round(m1_b1[0], 4)} \n Cgr = {round(m1_b1[1], 4)}'

    legend_entry = mlines.Line2D([], [], color='none', label=label_text)

    plt.plot(inst_g_r, new_std, label=f'Tgr = {round(m1_b1[0], 4)} \n Cgr = {round(m1_b1[1], 4)}')
    plt.scatter(inst_g_r, st_g_r)
    plt.xlabel('Instrumental (g-r)')       
    plt.ylabel('Standard (g-r)')       
    plt.title('Instrumental Color Index vs. Standard Color Index for July 31')
    plt.legend()
    plt.show()
    
    
    print(m1_b1[0], m1_b1[1], m2_b2[0], m2_b2[1])
    plt.plot(new_std, new_std_inst, label=f"Tg = {round(m2_b2[0], 4)}")
    plt.scatter(new_std, g_offset)
    plt.xlabel('Standard (g-r)')       
    plt.ylabel('Offset')       
    plt.title('Standard Color Index vs. Green Offset for July 31')
    plt.legend()
    plt.show()
    
    standard_g_r_target = m1_b1[0]*(target_g_inst_mag - target_r_inst_mag) + m1_b1[1]
    standard_g_target = m2_b2[0]*standard_g_r_target + m2_b2[1] + target_g_inst_mag
    


    standard_r_target = standard_g_target - standard_g_r_target

    error_g, error_r = m1_b1[2], m2_b2[2]
    Tgr = m1_b1[0]
    Cgr = m1_b1[1]
    Tg  = m2_b2[0]
    Cg  = m2_b2[1]

    show_three_labeled_stars("wcs_green_solution.fits", x_green, y_green, radius=5, n_show=3)
    show_three_labeled_stars("wcs_red_solution.fits", x_red, y_red, radius=5, n_show=3)





    
    print("\n===== DEBUG OUTPUT =====")
    print("Matched RA:", ra_match)
    print("Matched DEC:", dec_match)
    print("Green pixel coords (x, y):")
    print(x_green, y_green)
    print("Red pixel coords (x, y):")
    print(x_red, y_red)
    print("APASS g:", g)
    print("APASS r:", r)
    print("Green flux:", valid_flux_g)
    print("Red flux:", valid_flux_r)
    print("Instrumental g:", inst_g)
    print("Instrumental r:", inst_r)
    print("Instrumental (g-r):", inst_g_r)
    print("Standard (g-r):", st_g_r)
    print("Green offset:", g_offset)
    print("Tgr, Cgr:", Tgr, Cgr)
    print("Tg, Cg:", Tg, Cg)
    print("Standard error (color term):", m1_b1[2])
    print("Standard error (green offset):", m2_b2[2])
    print("========================\n")

    
    return (
        standard_g_target[0], 
        standard_r_target[0], 
        error_g, 
        error_r, 
        calibration_num, 
        Tgr, 
        Cgr,
        Tg,
        Cg
        ) 


full_calibration_with_subid("/Users/admin/YSPA-old-things/wcs-sn/r_july13-t24-mvn-green_stacked.fits", "wcs_green_solution.fits", os.environ.get("GREEN_SUBID")) # "/Users/admin/source/yspa-research-code/fits_files/r_july15-home40-mvn-green-all_stacked.fits"
num_rows = full_calibration_with_subid("/Users/admin/YSPA-old-things/wcs-sn/r_july13-t24-mvn-red_stacked.fits", "wcs_red_solution.fits", os.environ.get("RED_SUBID"))  #/Users/admin/source/yspa-research-code/fits_files/r_july15-home40-mvn-red-all_stacked.fits"
#full_calibration("/Users/admin/YSPA-old-things/wcs-sn/r_july13-t24-mvn-green_stacked.fits", "wcs_green_solution.fits") 
#num_rows = full_calibration("/Users/admin/YSPA-old-things/wcs-sn/r_july13-t24-mvn-red_stacked.fits", "wcs_red_solution.fits")

#full_calibration("/Users/admin/YSPA-old-things/wcs-sn/r_july15-home40-mvn-green-all_stacked.fits", "wcs_green_solution.fits") 
#num_rows = full_calibration("/Users/admin/YSPA-old-things/wcs-sn/r_july15-home40-mvn-red-all_stacked.fits", "wcs_red_solution.fits")
#num_rows = 15
print(magnitudes(
    "apass_subset.csv",
    "wcs_green_solution.fits",
    "wcs_red_solution.fits",
    num_rows,
    198.366170833, #198.49989,
    36.5865858333 #36.439874
))




