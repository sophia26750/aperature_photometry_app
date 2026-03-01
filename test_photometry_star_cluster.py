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
    radius = round(astro_results["radius"] * 0.6, 1)

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


def lsrl(x, y):
    x = np.array(x)
    y = np.array(y)

    if np.std(x) < 1e-3 or np.std(y) < 1e-3:
        raise ValueError("Not enough color variation for calibration.")

    m, b = np.polyfit(x, y, 1)
    sd = np.std(y - (m*x + b))
    return m, b, sd


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
    plt.show()

    return df_masked


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
    plt.show()

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
    plt.show()

    return df_cluster, df_calib


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
    """
    FINAL VERSION — NO MATCHING.
    - Uses APASS calibration stars from calibration_stars.csv
    - Detects ALL stars in the image
    - Splits detected stars by radius into cluster stars
    - Calibrates using APASS stars
    - Applies calibration to cluster stars
    - Produces calibrated CMD
    """

    # ============================================================
    # 1. Load APASS calibration stars (already masked)
    # ============================================================
    calib_cat = pd.read_csv("calibration_stars.csv")
    ap_ra  = calib_cat["ra"].values
    ap_dec = calib_cat["dec"].values
    g_std  = calib_cat["mag_g"].values
    r_std  = calib_cat["mag_r"].values

    # ============================================================
    # 2. Load WCS for green and red images
    # ============================================================
    w_g = WCS(green_image)
    w_r = WCS(red_image)

    # ============================================================
    # 3. Convert APASS calibration stars to pixel positions
    # ============================================================
    ap_x_g, ap_y_g = w_g.all_world2pix(ap_ra, ap_dec, 1)
    ap_x_r, ap_y_r = w_r.all_world2pix(ap_ra, ap_dec, 1)

    # ============================================================
    # 4. Measure flux for calibration stars
    # ============================================================
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

    # ============================================================
    # 5. Compute instrumental magnitudes for calibration stars
    # ============================================================
    inst_g = -2.5 * np.log10(green_flux)
    inst_r = -2.5 * np.log10(red_flux)

    inst_g_r = inst_g - inst_r
    st_g_r   = g_std - r_std
    g_offset = g_std - inst_g

    # ============================================================
    # 6. Solve calibration equations
    # ============================================================
    Tgr, Cgr, err_color = lsrl(inst_g_r, st_g_r)
    Tg,  Cg,  err_g     = lsrl(st_g_r, g_offset)

    print(f"Tgr={Tgr:.4f}, Cgr={Cgr:.4f}, Tg={Tg:.4f}, Cg={Cg:.4f}")
    print(f"σ_color={err_color:.4f}, σ_g={err_g:.4f}")

    # ============================================================
    # 7. Detect ALL stars in the green image
    # ============================================================
    sources = detect_stars(green_image, threshold=3.0)
    det_x = np.array(sources["xcentroid"])
    det_y = np.array(sources["ycentroid"])

    # Convert to RA/Dec
    det_ra, det_dec = w_g.all_pix2world(det_x, det_y, 1)
    det_coords = SkyCoord(det_ra*u.deg, det_dec*u.deg)
    center = SkyCoord(ra_center*u.deg, dec_center*u.deg)
    sep = det_coords.separation(center).arcminute

    # ============================================================
    # 8. Select cluster stars by radius
    # ============================================================
    cluster_mask = sep <= inner_radius_arcmin

    cl_x = det_x[cluster_mask]
    cl_y = det_y[cluster_mask]
    cl_ra = det_ra[cluster_mask]
    cl_dec = det_dec[cluster_mask]

    # ============================================================
    # 9. Measure flux for cluster stars
    # ============================================================
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

    # ============================================================
    # 10. Compute instrumental magnitudes for cluster stars
    # ============================================================
    cl_inst_g = -2.5 * np.log10(cl_flux_g)
    cl_inst_r = -2.5 * np.log10(cl_flux_r)
    cl_inst_g_r = cl_inst_g - cl_inst_r

    # ============================================================
    # 11. Apply calibration to cluster stars
    # ============================================================
    cl_std_g_r = Tgr * cl_inst_g_r + Cgr
    cl_std_g   = Tg * cl_std_g_r + Cg + cl_inst_g
    cl_std_r   = cl_std_g - cl_std_g_r

    # ============================================================
    # 12. Save calibrated cluster catalog
    # ============================================================
    out_df = pd.DataFrame({
        "ra": cl_ra,
        "dec": cl_dec,
        "inst_g": cl_inst_g,
        "inst_r": cl_inst_r,
        "std_g": cl_std_g,
        "std_r": cl_std_r,
        "std_g_r": cl_std_g_r
    })

    out_df.to_csv(output_calib_csv, index=False)
    print(f"Saved calibrated cluster catalog: {output_calib_csv}")


    print("Total detected stars:", len(det_x))
    print("Stars inside cluster radius:", np.sum(cluster_mask))
    print("Stars with good flux:", np.sum(good_cl))


    # ============================================================
    # 13. CMD plot
    # ============================================================
    plt.figure(figsize=(7,6))
    plt.scatter(cl_std_g_r, cl_std_g, s=12, alpha=0.8)
    plt.gca().invert_yaxis()
    plt.xlabel("Standard (g - r)")
    plt.ylabel("Standard g")
    plt.title("Cluster CMD (Calibrated)")
    plt.tight_layout()
    plt.show()

    # Color calibration
    new_std = Tgr * inst_g_r + Cgr
    plt.figure(figsize=(7,5))
    plt.plot(inst_g_r, new_std, label=f"Tgr = {Tgr:.4f}\nCgr = {Cgr:.4f}")
    plt.scatter(inst_g_r, st_g_r, s=15, alpha=0.7)
    plt.xlabel("Instrumental (g - r)")
    plt.ylabel("Standard (g - r)")
    plt.title("Instrumental vs Standard Color Index (Cluster)")
    plt.legend()
    plt.tight_layout()
    plt.show()

    # Offset vs color
    new_std_inst = Tg * new_std + Cg
    plt.figure(figsize=(7,5))
    plt.plot(new_std, new_std_inst, label=f"Tg = {Tg:.4f}")
    plt.scatter(new_std, g_offset, s=15, alpha=0.7)
    plt.xlabel("Standard (g - r)")
    plt.ylabel("g_offset = g_std - g_inst")
    plt.title("Standard Color Index vs Green Offset (Cluster)")
    plt.legend()
    plt.tight_layout()
    plt.show()

    return out_df



full_calibration_with_subid("/Users/admin/YSPA-old-things/starcluster-ngc6871/r_pp_green15-ngc6871.fits", "wcs_green_solution.fits", os.environ.get("GREEN_SUBID_SC")) 
num_rows = full_calibration_with_subid("/Users/admin/YSPA-old-things/starcluster-ngc6871/r_pp_red15-ngc6871.fits", "wcs_red_solution.fits", os.environ.get("RED_SUBID_SC")) 

#full_calibration("/Users/admin/YSPA-old-things/starcluster-ngc6871/r_pp_green15-ngc6871.fits", "wcs_green_solution.fits") 
#num_rows = full_calibration("/Users/admin/YSPA-old-things/starcluster-ngc6871/r_pp_red15-ngc6871.fits", "wcs_red_solution.fits")

inner_radius_arcmin = 20      # inner radius
outer_radius_arcmin = inner_radius_arcmin * 1.5      # outer radius


split_apass_cluster_and_calibration(
    "apass_subset.csv",
    ra_center=301.7333,
    dec_center=35.8492,
    cluster_radius_arcmin=inner_radius_arcmin,
    outer_radius_arcmin=outer_radius_arcmin
)

show_cluster_and_calibration_image(
    "wcs_green_solution.fits",
    ra_center=301.7333,
    dec_center=35.8492,
    cluster_radius_arcmin=inner_radius_arcmin,
    cluster_csv="cluster_stars.csv",
    calib_csv="calibration_stars.csv"
)


print(star_cluster_magnitudes(
    "apass_subset.csv",
    "wcs_green_solution.fits",
    "wcs_red_solution.fits",
    ra_center=301.7333,
    dec_center=35.8492,
    inner_radius_arcmin=inner_radius_arcmin,
    outer_radius_arcmin=outer_radius_arcmin
))



