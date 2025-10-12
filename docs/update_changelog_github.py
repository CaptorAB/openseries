"""Update changelog with latest GitHub release information."""

import json
import re
from pathlib import Path
from urllib.request import urlopen
from urllib.error import HTTPError, URLError


def fetch_latest_release():
    """Fetch the latest release information from GitHub API."""
    api_url = "https://api.github.com/repos/CaptorAB/openseries/releases/latest"

    try:
        with urlopen(api_url) as response:
            data = json.loads(response.read().decode())
            return {
                "version": data["tag_name"],
                "name": data["name"],
                "body": data["body"],
                "html_url": data["html_url"],
                "published_at": data["published_at"],
            }
    except (HTTPError, URLError) as e:
        print(f"Warning: Could not fetch latest release from GitHub: {e}")
        return None


def update_changelog():
    """Update the changelog with the latest GitHub release information."""
    changelog_path = Path("source/development/changelog.rst")

    # Read the current changelog
    with changelog_path.open(mode="r", encoding="utf-8") as f:
        content = f.read()

    # Fetch latest release info
    release_info = fetch_latest_release()

    if release_info:
        version = release_info["version"]
        release_url = release_info["html_url"]

        # Extract description from the release body
        body = release_info["body"]
        if body:
            # Clean up the body text (remove markdown formatting, etc.)
            # Replace \r\n with spaces and clean up
            clean_body = body.replace("\r\n", " ").replace("\n", " ").strip()

            # Remove markdown list markers and clean up
            clean_body = re.sub(r"^-\s*", "", clean_body)  # Remove leading "- "
            clean_body = re.sub(r"\s+", " ", clean_body)  # Normalize whitespace

            # Limit length for display
            if len(clean_body) > 120:
                description = clean_body[:117] + "..."
            else:
                description = clean_body
        else:
            description = "See GitHub release for details"

        # Replace the version placeholders
        updated_content = content.replace("|current_version|", version)

        # Update the current release description
        # Find the first line with "Version {version}" (after replacement) and update it
        pattern = rf"(\*\*Version {re.escape(version)}\*\*: ).*"
        replacement = f"\\1{description}"
        updated_content = re.sub(
            pattern, replacement, updated_content, count=1
        )  # Only replace first occurrence

        print(f"Updated changelog with latest GitHub release: {version}")
    else:
        # Fallback: use a generic message
        updated_content = content.replace("|current_version|", "latest")
        print("Could not fetch latest release, using fallback")

    # Write the updated content back
    with changelog_path.open(mode="w", encoding="utf-8") as f:
        f.write(updated_content)


if __name__ == "__main__":
    update_changelog()
