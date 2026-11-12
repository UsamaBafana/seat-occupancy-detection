"""
list_cameras.py
Instantly print which index maps to which physical camera, by name.

Windows can silently reorder USB camera indices between separate
process launches (often from USB selective suspend). The device NAME
never changes though, so checking by name is faster and more reliable
than grabbing a snapshot from each index every time.

Requires: pip install pygrabber

Usage:
  python scripts/list_cameras.py
"""
from pygrabber.dshow_graph import FilterGraph


def main() -> None:
    graph = FilterGraph()
    devices = graph.get_input_devices()

    if not devices:
        print("No cameras found.")
        return

    print("\nCurrent camera index -> name mapping:")
    for i, name in enumerate(devices):
        print(f"  {i}: {name}")
    print(
        "\nUse the index shown here for --source / --sources — "
        "this mapping is what cv2.VideoCapture(i, cv2.CAP_DSHOW) will "
        "actually open right now, in this session.\n"
    )


if __name__ == "__main__":
    main()
