"""Is the HAA loudspeaker directional, and which way does it face?  For each GT RIR: locate the
direct-path arrival, take a short window, compute band energies (low <500 Hz vs high 2-8 kHz),
distance-normalise (x r^2), and regress against the receiver azimuth about the speaker."""
import json, numpy as np, soundfile as sf, os, sys
root='/media/diskstation/yixunhu/FLAC/HAA'
sc=json.load(open(root+'/metadata/scenes_metadata.json')); po=json.load(open(root+'/metadata/poses_metadata.json'))
sr=22050
def band_energy(x, lo, hi):
    X=np.fft.rfft(x); f=np.fft.rfftfreq(len(x), 1/sr); m=(f>=lo)&(f<hi); return float(np.sum(np.abs(X[m])**2))
rooms = sys.argv[1:] or list(sc)
for s in rooms:
    spk=np.array(sc[s]['speaker_xyz']); rows=[]
    ids = sorted(int(k) for k in po[s].keys())
    for rid in ids[::2]:   # every other receiver (speed)
        fp=f'{root}/{s}/mono_rirs_22050Hz/{rid}.wav'
        if not os.path.exists(fp): continue
        x,r=sf.read(fp); x=x if x.ndim==1 else x[:,0]
        p=np.array(po[s][str(rid)])-spk; d=np.linalg.norm(p); az=np.degrees(np.arctan2(p[1],p[0]))
        # direct arrival: first sample exceeding 30% of the peak within the first 60 ms
        env=np.abs(x); pk=env.max(); i0=int(np.argmax(env>0.3*pk)); w=x[i0:i0+int(0.0025*sr)]
        if len(w)<40: continue
        lo=band_energy(w,80,500); hi=band_energy(w,2000,8000); tot=band_energy(w,80,8000)
        rows.append((az,d,p[0],p[1],lo*d*d,hi*d*d,tot*d*d))
    a=np.array(rows)
    print(f"== {s}: n={len(a)} speaker {spk.round(2)}")
    # bin by azimuth (8 sectors); report mean log10 of distance-corrected energies
    bins=np.linspace(-180,180,9); idx=np.digitize(a[:,0],bins)-1
    print("   sector(deg)   n   log10(low*r2)  log10(high*r2)  high/low(dB)   log10(tot*r2)")
    for b in range(8):
        m=idx==b
        if m.sum()==0: continue
        lo=np.log10(a[m,4]).mean(); hi=np.log10(a[m,5]).mean(); tot=np.log10(a[m,6]).mean()
        print(f"   [{bins[b]:6.0f},{bins[b+1]:6.0f})  {m.sum():3d}   {lo:7.2f}        {hi:7.2f}        {10*(hi-lo):6.1f}       {tot:7.2f}")
    # hallway-style: split by sign of y and x
    for ax,name in ((1,'y'),(0,'x')):
        pos=a[:,2+ax]>0; neg=~pos
        if pos.sum()>3 and neg.sum()>3:
            print(f"   {name}>0 (n={pos.sum()}): high*r2 {np.log10(a[pos,5]).mean():.2f}  tot*r2 {np.log10(a[pos,6]).mean():.2f} | {name}<0 (n={neg.sum()}): high*r2 {np.log10(a[neg,5]).mean():.2f}  tot*r2 {np.log10(a[neg,6]).mean():.2f}")
