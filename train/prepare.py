"""Download tiny-shakespeare (karpathy/char-rnn) into data/input.txt and check its sha256."""
import hashlib, pathlib, urllib.request

URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
SHA256 = "86c4e6aa9db7c042ec79f339dcb96d42b0075e16b8fc2e86bf0ca57e2dc565ed"  # the file these models were trained on
out = pathlib.Path(__file__).resolve().parent.parent / "data" / "input.txt"
out.parent.mkdir(exist_ok=True)
if not out.exists():
    urllib.request.urlretrieve(URL, out)
data = out.read_bytes()
digest = hashlib.sha256(data).hexdigest()
print(len(data), "bytes, sha256", digest)
assert digest == SHA256, "data/input.txt is not the file the models were trained on"
