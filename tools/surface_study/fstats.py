import sys, numpy as np
for name in sys.argv[1:]:
    p = np.load(f"/vesuvius/usrm2/volcomp_surface/float/{name}.npy", mmap_mode="r")
    a = np.asarray(p[:, :512, :512])
    u = np.clip(np.rint(a * 255), 0, 255).astype(np.uint8)
    h = np.bincount(u.ravel(), minlength=256) / u.size
    print(name, "min %.3g max %.3g" % (a.min(), a.max()), "frac p==0 %.4f  p<1e-3 %.4f  u8==0 %.4f u8==255 %.4f  u8 1..254 %.4f  >=128 %.4f" % (
        (a == 0).mean(), (a < 1e-3).mean(), h[0], h[255], h[1:255].sum(), h[128:].sum()))
    print("  hist/16", " ".join("%.4f" % h[i:i+16].sum() for i in range(0, 256, 16)))
    print("  hist 0..8", " ".join("%.4f" % x for x in h[:9]), " 247..255", " ".join("%.4f" % x for x in h[247:]))
    # band thickness: distance from the 0.5 surface for voxels with 0.02<p<0.98 via EDT
    from scipy import ndimage as ndi
    m = u >= 128
    d = np.where(m, ndi.distance_transform_edt(m), ndi.distance_transform_edt(~m))
    s = (u > 2) & (u < 253)
    print("  soft voxels (2<u<253): %.4f; their dist to 0.5 surf pctl 50/90/99: %s" % (s.mean(), np.percentile(d[s], [50, 90, 99]).round(2)))
    for k in (1, 2, 3, 4, 6, 8, 12):
        print("   within %2d vox of surf: frac %.4f  covers %.4f of soft voxels" % (k, (d <= k).mean(), (s & (d <= k)).sum() / max(s.sum(), 1)))
