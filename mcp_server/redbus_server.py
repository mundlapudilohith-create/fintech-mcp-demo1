"""
RedBus Redirect MCP Server
Generates deep links and web URLs to redirect users to RedBus
app or website with pre-filled search parameters.

Tools:
  - redbus_search_redirect     → search buses between two cities
  - redbus_booking_redirect    → redirect to a specific bus booking
  - redbus_offers_redirect     → redirect to offers/deals page
  - redbus_tracking_redirect   → redirect to ticket tracking page
  - get_popular_routes         → list popular routes from a city
"""

from fastmcp import FastMCP
from typing import Optional
from datetime import datetime, date
import logging
import urllib.parse

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

mcp = FastMCP("RedBus Redirect")

# ── Base URLs ──────────────────────────────────────────────────────────────────
REDBUS_WEB_BASE   = "https://www.redbus.in"
REDBUS_APP_SCHEME = "redbus://"          # Android/iOS deep link scheme
REDBUS_APP_WEB    = "https://app.redbus.in"  # Universal link fallback

# ── Popular cities (for validation + suggestions) ─────────────────────────────
POPULAR_CITIES = [
    "Bangalore", "Mumbai", "Delhi", "Chennai", "Hyderabad",
    "Pune", "Kolkata", "Ahmedabad", "Jaipur", "Surat",
    "Kochi", "Coimbatore", "Visakhapatnam", "Nagpur", "Indore",
    "Bhopal", "Chandigarh", "Lucknow", "Patna", "Goa"
]


# ── Tool 1: Bus Search Redirect ────────────────────────────────────────────────
@mcp.tool()
def redbus_search_redirect(
    source_city: str,
    destination_city: str,
    travel_date: Optional[str] = None,
    redirect_to: str = "web"
) -> dict:
    """
    Generate a RedBus search URL to redirect user to bus search results.

    Args:
        source_city     : Departure city (e.g., "Bangalore")
        destination_city: Arrival city (e.g., "Mumbai")
        travel_date     : Date in YYYY-MM-DD format (defaults to today)
        redirect_to     : "web" for browser, "app" for mobile app deep link,
                          "both" for web URL + app deep link

    Returns:
        Dictionary with redirect URLs and display message

    Example:
        redbus_search_redirect("Bangalore", "Mumbai", "2026-03-01", "web")
    """
    logger.info(f"RedBus search: {source_city} → {destination_city} on {travel_date}")

    # Default to today if no date provided
    if not travel_date:
        travel_date = date.today().strftime("%Y-%m-%d")

    # Format date for RedBus URL: DD-Mon-YYYY (e.g., 01-Mar-2026)
    try:
        dt = datetime.strptime(travel_date, "%Y-%m-%d")
        redbus_date = dt.strftime("%d-%b-%Y")  # 01-Mar-2026
    except ValueError:
        redbus_date = datetime.today().strftime("%d-%b-%Y")

    # Encode city names for URL
    src_encoded  = urllib.parse.quote(source_city.strip())
    dst_encoded  = urllib.parse.quote(destination_city.strip())

    # Web URL: https://www.redbus.in/bus-tickets/bangalore-to-mumbai?doj=01-Mar-2026
    route_slug = f"{source_city.lower().replace(' ', '-')}-to-{destination_city.lower().replace(' ', '-')}"
    web_url    = f"{REDBUS_WEB_BASE}/bus-tickets/{route_slug}?doj={redbus_date}"

    # App deep link: redbus://search?src=Bangalore&dst=Mumbai&doj=01-Mar-2026
    app_params   = urllib.parse.urlencode({
        "src": source_city,
        "dst": destination_city,
        "doj": redbus_date
    })
    app_deeplink = f"{REDBUS_APP_SCHEME}search?{app_params}"

    # Universal link (works if app installed, falls back to web)
    universal_url = f"{REDBUS_APP_WEB}/bus-tickets/{route_slug}?doj={redbus_date}"

    result = {
        "action":           "search_buses",
        "source":           source_city,
        "destination":      destination_city,
        "travel_date":      travel_date,
        "display_date":     redbus_date,
        "web_url":          web_url,
        "app_deeplink":     app_deeplink,
        "universal_url":    universal_url,
        "message":          f"Search buses from {source_city} to {destination_city} on {redbus_date}",
        "redirect_message": f"Click here to view available buses → {web_url}"
    }

    if redirect_to == "app":
        result["redirect_url"] = app_deeplink
        result["fallback_url"] = web_url
    elif redirect_to == "both":
        result["redirect_url"] = universal_url
        result["fallback_url"] = web_url
    else:
        result["redirect_url"] = web_url

    logger.info(f"RedBus URL generated: {result['redirect_url']}")
    return result


# ── Tool 2: Booking Redirect ───────────────────────────────────────────────────
@mcp.tool()
def redbus_booking_redirect(
    tin: str,
    redirect_to: str = "web"
) -> dict:
    """
    Generate a RedBus URL to view/manage an existing booking.

    Args:
        tin         : Ticket ID / TIN number from RedBus booking confirmation
        redirect_to : "web" or "app"

    Returns:
        Dictionary with booking management URL

    Example:
        redbus_booking_redirect("TIN123456789")
    """
    logger.info(f"RedBus booking redirect: TIN={tin}")

    tin_clean = tin.strip().upper()

    web_url      = f"{REDBUS_WEB_BASE}/mybookings/ticket-details?tin={tin_clean}"
    app_deeplink = f"{REDBUS_APP_SCHEME}booking?tin={tin_clean}"

    result = {
        "action":        "view_booking",
        "tin":           tin_clean,
        "web_url":       web_url,
        "app_deeplink":  app_deeplink,
        "redirect_url":  app_deeplink if redirect_to == "app" else web_url,
        "message":       f"View your RedBus booking (TIN: {tin_clean})",
        "redirect_message": f"Click here to view your booking → {web_url}"
    }

    logger.info(f"Booking URL: {result['redirect_url']}")
    return result


# ── Tool 3: Offers Redirect ────────────────────────────────────────────────────
@mcp.tool()
def redbus_offers_redirect(
    source_city: Optional[str] = None,
    redirect_to: str = "web"
) -> dict:
    """
    Generate a RedBus URL to view current offers and deals.

    Args:
        source_city : Optional city to filter offers (e.g., "Bangalore")
        redirect_to : "web" or "app"

    Returns:
        Dictionary with offers page URL

    Example:
        redbus_offers_redirect("Bangalore", "web")
    """
    logger.info(f"RedBus offers redirect: city={source_city}")

    web_url = f"{REDBUS_WEB_BASE}/offers"
    if source_city:
        web_url += f"?src={urllib.parse.quote(source_city)}"

    app_deeplink = f"{REDBUS_APP_SCHEME}offers"
    if source_city:
        app_deeplink += f"?src={urllib.parse.quote(source_city)}"

    city_text = f" from {source_city}" if source_city else ""

    result = {
        "action":           "view_offers",
        "source_city":      source_city,
        "web_url":          web_url,
        "app_deeplink":     app_deeplink,
        "redirect_url":     app_deeplink if redirect_to == "app" else web_url,
        "message":          f"View RedBus offers and deals{city_text}",
        "redirect_message": f"Click here to view offers → {web_url}"
    }

    logger.info(f"Offers URL: {result['redirect_url']}")
    return result


# ── Tool 4: Ticket Tracking Redirect ──────────────────────────────────────────
@mcp.tool()
def redbus_tracking_redirect(
    tin: str,
    redirect_to: str = "web"
) -> dict:
    """
    Generate a RedBus URL to track a live bus journey.

    Args:
        tin         : Ticket ID / TIN number from booking confirmation
        redirect_to : "web" or "app"

    Returns:
        Dictionary with live tracking URL

    Example:
        redbus_tracking_redirect("TIN123456789")
    """
    logger.info(f"RedBus tracking redirect: TIN={tin}")

    tin_clean    = tin.strip().upper()
    web_url      = f"{REDBUS_WEB_BASE}/mybookings/track-my-bus?tin={tin_clean}"
    app_deeplink = f"{REDBUS_APP_SCHEME}tracking?tin={tin_clean}"

    result = {
        "action":           "track_bus",
        "tin":              tin_clean,
        "web_url":          web_url,
        "app_deeplink":     app_deeplink,
        "redirect_url":     app_deeplink if redirect_to == "app" else web_url,
        "message":          f"Track your live bus journey (TIN: {tin_clean})",
        "redirect_message": f"Click here to track your bus → {web_url}"
    }

    logger.info(f"Tracking URL: {result['redirect_url']}")
    return result


# ── Tool 5: Popular Routes ─────────────────────────────────────────────────────
@mcp.tool()
def get_popular_routes(
    source_city: Optional[str] = None
) -> dict:
    """
    Get popular RedBus routes from a given city or overall top routes.

    Args:
        source_city: Optional source city to filter routes (e.g., "Bangalore")

    Returns:
        Dictionary with popular routes and their RedBus URLs

    Example:
        get_popular_routes("Bangalore")
    """
    logger.info(f"Popular routes: source={source_city}")

    # Popular routes database
    all_routes = {
        "Bangalore": ["Mumbai", "Chennai", "Hyderabad", "Pune", "Goa", "Coimbatore", "Kochi"],
        "Mumbai":    ["Pune", "Goa", "Bangalore", "Ahmedabad", "Nashik", "Shirdi"],
        "Delhi":     ["Jaipur", "Agra", "Chandigarh", "Lucknow", "Haridwar", "Shimla"],
        "Chennai":   ["Bangalore", "Coimbatore", "Hyderabad", "Kochi", "Madurai", "Pondicherry"],
        "Hyderabad": ["Bangalore", "Chennai", "Mumbai", "Pune", "Vijayawada", "Vizag"],
        "Pune":      ["Mumbai", "Goa", "Bangalore", "Nashik", "Kolhapur", "Shirdi"],
    }

    today = date.today().strftime("%d-%b-%Y")

    if source_city and source_city in all_routes:
        destinations = all_routes[source_city]
        routes = []
        for dst in destinations:
            slug = f"{source_city.lower()}-to-{dst.lower()}"
            routes.append({
                "from":    source_city,
                "to":      dst,
                "web_url": f"{REDBUS_WEB_BASE}/bus-tickets/{slug}?doj={today}",
                "label":   f"{source_city} → {dst}"
            })
        message = f"Popular routes from {source_city}"
    else:
        # Top overall routes
        routes = [
            {"from": "Bangalore", "to": "Chennai",   "web_url": f"{REDBUS_WEB_BASE}/bus-tickets/bangalore-to-chennai?doj={today}",   "label": "Bangalore → Chennai"},
            {"from": "Mumbai",    "to": "Pune",       "web_url": f"{REDBUS_WEB_BASE}/bus-tickets/mumbai-to-pune?doj={today}",         "label": "Mumbai → Pune"},
            {"from": "Delhi",     "to": "Jaipur",     "web_url": f"{REDBUS_WEB_BASE}/bus-tickets/delhi-to-jaipur?doj={today}",        "label": "Delhi → Jaipur"},
            {"from": "Hyderabad", "to": "Bangalore",  "web_url": f"{REDBUS_WEB_BASE}/bus-tickets/hyderabad-to-bangalore?doj={today}", "label": "Hyderabad → Bangalore"},
            {"from": "Chennai",   "to": "Coimbatore", "web_url": f"{REDBUS_WEB_BASE}/bus-tickets/chennai-to-coimbatore?doj={today}",  "label": "Chennai → Coimbatore"},
            {"from": "Pune",      "to": "Goa",        "web_url": f"{REDBUS_WEB_BASE}/bus-tickets/pune-to-goa?doj={today}",            "label": "Pune → Goa"},
        ]
        message = "Top popular routes on RedBus"

    return {
        "action":        "popular_routes",
        "source_city":   source_city,
        "routes":        routes,
        "total_routes":  len(routes),
        "popular_cities": POPULAR_CITIES,
        "message":       message
    }



# ── Tool 6: Open RedBus (Homepage / App) ──────────────────────────────────────
@mcp.tool()
def open_redbus(
    redirect_to: str = "web"
) -> dict:
    """
    Open RedBus homepage directly — app or website.
    Use this when user says "open redbus", "go to redbus", "launch redbus",
    "redbus app", or any general intent to open/visit RedBus.

    Args:
        redirect_to: "web"  → opens https://www.redbus.in  (default)
                     "app"  → opens RedBus mobile app via deep link
                     "both" → returns both URLs

    Returns:
        Dictionary with redirect URL and display message

    Examples:
        open_redbus()           → web homepage
        open_redbus("app")      → mobile app deep link
        open_redbus("both")     → both URLs
    """
    logger.info(f"Open RedBus: redirect_to={redirect_to}")

    web_url      = REDBUS_WEB_BASE                  # https://www.redbus.in
    app_deeplink = f"{REDBUS_APP_SCHEME}home"        # redbus://home
    play_store   = "https://play.google.com/store/apps/details?id=in.redbus.android"
    app_store    = "https://apps.apple.com/in/app/redbus-bus-ticket-booking/id539179365"

    if redirect_to == "app":
        redirect_url   = app_deeplink
        fallback_url   = web_url
        display_action = "Opening RedBus app..."
    elif redirect_to == "both":
        redirect_url   = web_url
        fallback_url   = app_deeplink
        display_action = "Opening RedBus..."
    else:
        redirect_url   = web_url
        fallback_url   = None
        display_action = "Opening RedBus website..."

    result = {
        "action":          "open_redbus",
        "redirect_url":    redirect_url,
        "web_url":         web_url,
        "app_deeplink":    app_deeplink,
        "fallback_url":    fallback_url,
        "play_store_url":  play_store,
        "app_store_url":   app_store,
        "message":         display_action,
        "redirect_message": f"Click here to open RedBus → {redirect_url}",
        "popular_cities":  POPULAR_CITIES[:8],
        "tip":             "You can also say 'book bus from Bangalore to Mumbai' to search directly!"
    }

    logger.info(f"Open RedBus URL: {redirect_url}")
    return result

# ── Entry point ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    logger.info("Starting RedBus Redirect MCP Server...")
    mcp.run()