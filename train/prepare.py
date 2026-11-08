"""Download tiny-shakespeare (karpathy/char-rnn) into data/input.txt and check it."""
import hashlib, pathlib, urllib.request

URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
out = pathlib.Path(__file__).resolve().parent.parent / "data" / "input.txt"
out.parent.mkdir(exist_ok=True)
if not out.exists():
    urllib.request.urlretrieve(URL, out)
data = out.read_bytes()
print(len(data), "bytes, sha256", hashlib.sha256(data).hexdigest())
