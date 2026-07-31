"""S3 extension: fail-closed distilled-backbone loading (plan §5 's3')."""
import hashlib, types, pytest, torch
from conftest import load_mod

ib = load_mod("init_backbone")


class CylindricalDINOv3ViTModel(torch.nn.Module):     # name is what the check reads
    def __init__(self):
        super().__init__()
        self.w = torch.nn.Parameter(torch.zeros(2, 2))


def fake_model(shared=True, cls=CylindricalDINOv3ViTModel):
    vit = cls()
    m = types.SimpleNamespace()
    src = types.SimpleNamespace(vit=vit)
    ctx = types.SimpleNamespace(vit=vit if shared else cls())
    m.conditioner = types.SimpleNamespace(conditioners={"source_vit": src,
                                                        "context_poses_vit": ctx})
    return m, vit


def artifact(tmp_path, sd):
    p = tmp_path / "bb.pt"
    torch.save(sd, str(p))
    sha = hashlib.sha256(open(p, "rb").read()).hexdigest()
    return str(p), sha


def test_happy_load(tmp_path):
    m, vit = fake_model()
    p, sha = artifact(tmp_path, {"w": torch.ones(2, 2)})
    n, bb = ib.load_distilled_backbone(m, p, sha)
    assert n == 4 and bb is vit and torch.equal(vit.w, torch.ones(2, 2))


def test_sha_mismatch_refused(tmp_path):
    m, _ = fake_model()
    p, _ = artifact(tmp_path, {"w": torch.ones(2, 2)})
    with pytest.raises(SystemExit) as e:
        ib.load_distilled_backbone(m, p, "0" * 64)
    assert "sha" in str(e.value)


def test_unshared_backbone_refused(tmp_path):
    m, _ = fake_model(shared=False)
    p, sha = artifact(tmp_path, {"w": torch.ones(2, 2)})
    with pytest.raises(SystemExit) as e:
        ib.load_distilled_backbone(m, p, sha)
    assert "same object" in str(e.value)


def test_wrong_class_refused(tmp_path):
    class OtherViT(CylindricalDINOv3ViTModel):
        pass
    m, _ = fake_model(cls=OtherViT)
    p, sha = artifact(tmp_path, {"w": torch.ones(2, 2)})
    with pytest.raises(SystemExit) as e:
        ib.load_distilled_backbone(m, p, sha)
    assert "backbone class" in str(e.value)


def test_strict_key_mismatch_refused(tmp_path):
    m, _ = fake_model()
    p, sha = artifact(tmp_path, {"wrong_key": torch.ones(2, 2)})
    with pytest.raises(SystemExit) as e:
        ib.load_distilled_backbone(m, p, sha)
    assert "strict load" in str(e.value)


def test_missing_conditioners_refused(tmp_path):
    p, sha = artifact(tmp_path, {"w": torch.ones(2, 2)})
    with pytest.raises(SystemExit) as e:
        ib.load_distilled_backbone(types.SimpleNamespace(), p, sha)
    assert "expected conditioner" in str(e.value)
