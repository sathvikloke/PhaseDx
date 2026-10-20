"""Three retained B31 repair paths against the portable module; fake storage/HTTP."""
import datetime
import email.message
import hashlib
import io
import json
import urllib.request
import urllib.response

import pytest

from phasedx_external import acquire31 as A

DATA = bytes((i * 7 + 3) % 256 for i in range(640))


class Body(io.BytesIO):
    def __init__(self, data):
        super().__init__(data)
        self.received = 0

    def read(self, n=-1):
        b = super().read(n)
        self.received += len(b)
        return b


class Fake(urllib.request.BaseHandler):
    handler_order = 100

    def __init__(self):
        self.requests = []
        self.body = None

    def https_open(self, req):
        self.requests.append(req.full_url)
        self.body = Body(DATA)
        msg = email.message.Message()
        msg["Content-Length"] = str(len(DATA))
        msg["ETag"] = '"' + hashlib.md5(DATA).hexdigest() + '"'
        response = urllib.response.addinfourl(self.body, msg, req.full_url, 200)
        response.msg = "OK"
        return response


def setup(tmp_path, **options):
    deadline = (datetime.datetime.now(datetime.timezone.utc) + datetime.timedelta(minutes=5)).isoformat()
    ctx = A.OpContext(str(tmp_path/"data"), deadline, [], baseline_bytes=0, min_free=0, **options)
    ctx.acquire()
    row = {"name": "fake.zip", "url": "https://fixture.invalid/fake.zip", "size_bytes": len(DATA),
           "expected_md5_from_object_metadata": hashlib.md5(DATA).hexdigest(),
           "expected_sha256": "", "etag": hashlib.md5(DATA).hexdigest(), "role": "images"}
    return ctx, row


def test_received_bytes_count_even_when_space_disappears_before_write(tmp_path):
    h = Fake()
    c, row = setup(tmp_path, free_fn=lambda p: 0 if h.body and h.body.received >= 192 else 10**12)
    try:
        with pytest.raises(A.Stopped, match="insufficient_space_runtime"):
            A.fetch_one(row, c, "FIXTURE", {"fixture.invalid"},
                        urllib.request.build_opener(A.NoRedirect, h), chunk=64)
        part = tmp_path/"data/archives/fake.zip.part"
        assert part.read_bytes() == DATA[:128]
        assert [x["bytes"] for x in c.events() if x["kind"] == "settle"] == [192]
    finally:
        c.release()


def test_probe_is_reserved_before_exact_budget_request(tmp_path):
    h = Fake()
    c, row = setup(tmp_path, cumulative_cap=len(DATA))
    try:
        with pytest.raises(A.Refuse, match="cumulative cap"):
            A.fetch_one(row, c, "FIXTURE", {"fixture.invalid"},
                        urllib.request.build_opener(A.NoRedirect, h))
        assert h.requests == []
    finally:
        c.release()


def test_complete_part_is_verified_and_promoted_without_http(tmp_path):
    h = Fake()
    c, row = setup(tmp_path)
    try:
        folder = tmp_path/"data/archives"
        folder.mkdir(exist_ok=True)
        (folder/"fake.zip.part").write_bytes(DATA)
        (folder/"fake.zip.part.json").write_text(json.dumps({
            "source_list_sha256": "FIXTURE", "url": row["url"], "size_bytes": len(DATA),
            "expected_md5_from_object_metadata": hashlib.md5(DATA).hexdigest(),
            "etag_listed": hashlib.md5(DATA).hexdigest(), "server": {}}))
        A.fetch_one(row, c, "FIXTURE", {"fixture.invalid"},
                    urllib.request.build_opener(A.NoRedirect, h))
        assert (folder/"fake.zip").read_bytes() == DATA
        assert not (folder/"fake.zip.part").exists()
        assert h.requests == []
    finally:
        c.release()
