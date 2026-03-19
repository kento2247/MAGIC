import argparse
import base64
import json
import os
import shutil
import subprocess
import tempfile
import xml.etree.ElementTree as ET
from dataclasses import dataclass, asdict
from io import BytesIO
from pathlib import Path
from typing import Any

from PIL import Image

