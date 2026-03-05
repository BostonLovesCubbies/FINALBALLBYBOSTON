import streamlit as st
import pandas as pd
import numpy as np
import math
import re
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patheffects as pe
from matplotlib.patches import Rectangle, Polygon, FancyBboxPatch
from matplotlib.colors import LinearSegmentedColormap
from scipy.stats import gaussian_kde
from io import BytesIO
import warnings
warnings.filterwarnings("ignore")

st.set_page_config(page_title="TrackMan Cards", page_icon="📡", layout="wide")
st.title("📡 TrackMan Pitcher Cards")

# ── Constants (same as MLB card) ──────────────────────────────────────────────
PITCH_COLORS = {
    'FF':'#D22D49','SI':'#C57A02','FC':'#933F2C','SL':'#DDB33A',
    'ST':'#A2C8FC','CU':'#00D1ED','KC':'#3BACAC','CH':'#1DBE3A',
    'FS':'#3CB84B','SV':'#9B59B6','KN':'#888888','EP':'#aaaaaa',
    'OTHER':'#777777',
}
PITCH_NAMES = {
    'FF':'4-Seam Fastball','SI':'Sinker','FC':'Cutter','SL':'Slider',
    'ST':'Sweeper','CU':'Curveball','KC':'Knuckle Curve','CH':'Changeup',
    'FS':'Splitter','SV':'Slurve','KN':'Knuckleball','EP':'Eephus','OTHER':'Other',
}
# TrackMan TaggedPitchType → our codes
TM_TYPE_MAP = {
    'Fastball':'FF','Four-Seam':'FF','FourSeamFastBall':'FF','4-Seam':'FF',
    'Sinker':'SI','TwoSeamFastBall':'SI','Two-Seam':'SI',
    'Cutter':'FC','Cut':'FC',
    'Slider':'SL','Sweeper':'ST',
    'Curveball':'CU','CurveBall':'CU','12-6 Curveball':'CU',
    'Knuckle Curve':'KC','KnuckleCurve':'KC',
    'ChangeUp':'CH','Changeup':'CH','Change':'CH',
    'Splitter':'FS','Split-Finger':'FS',
    'Knuckleball':'KN',
    'Eephus':'EP',
    'Undefined':'OTHER','Other':'OTHER','AutomaticBall':'OTHER',
}

HEATMAP_COLORS=['#0a0a2e','#1a1a6e','#2244aa','#4477cc','#77aaee',
                '#aaccff','#ffffff','#ffcc66','#ff8833','#ee3311','#aa0000']
SAVANT_CMAP=LinearSegmentedColormap.from_list('savant',HEATMAP_COLORS,N=256)
BG_COLOR='#0d0d0d'; PANEL_COLOR='#161616'; BORDER_COLOR='#2a2a2a'
TEXT_COLOR='#e8e8e8'; DIM_COLOR='#aaaaaa'

# ── Column detection ──────────────────────────────────────────────────────────
def find_col(df, candidates):
    """Case-insensitive column finder."""
    cols_lower = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand.lower() in cols_lower:
            return cols_lower[cand.lower()]
    return None

def detect_columns(df):
    """Map standard names to actual TrackMan column names."""
    return {
        'pitcher':    find_col(df, ['Pitcher','pitcher','PitcherName']),
        'throws':     find_col(df, ['PitcherThrows','pitcherThrows','Throws','Hand']),
        'pitch_type': find_col(df, ['TaggedPitchType','AutoPitchType','PitchType','pitch_type']),
        'velo':       find_col(df, ['RelSpeed','releaseSpeed','Velo','velocity','StartSpeed']),
        'spin':       find_col(df, ['SpinRate','spinRate','Spin']),
        'spin_axis':  find_col(df, ['SpinAxis','spinAxis']),
        'hb':         find_col(df, ['HorzBreak','horzBreak','HorizontalBreak','pfx_x']),
        'ivb':        find_col(df, ['InducedVertBreak','inducedVertBreak','InducedVB','pfx_z']),
        'ext':        find_col(df, ['Extension','extension','ReleaseExtension']),
        'plate_x':    find_col(df, ['PlateLocSide','plateLocSide','plate_x','PlateX']),
        'plate_z':    find_col(df, ['PlateLocHeight','plateLocHeight','plate_z','PlateZ']),
        'batter_side':find_col(df, ['BatterSide','batterSide','Stand','stand','BatterHand']),
        'pitch_call': find_col(df, ['PitchCall','pitchCall','description','Result']),
        'exit_speed': find_col(df, ['ExitSpeed','exitSpeed','launch_speed','ExitVelocity']),
        'launch_angle':find_col(df, ['Angle','angle','LaunchAngle','launch_angle','VertAngle']),
        'play_result':find_col(df, ['PlayResult','playResult','events','KorBB']),
        'game_date':  find_col(df, ['Date','GameDate','game_date','Date']),
    }

def map_pitch_types(df, col):
    """Map TrackMan pitch type strings to our short codes."""
    if col is None: return pd.Series(['OTHER']*len(df), index=df.index)
    mapped = df[col].astype(str).map(lambda x: TM_TYPE_MAP.get(x, TM_TYPE_MAP.get(x.strip(), 'OTHER')))
    return mapped

def _pct(num, den): return (num/den*100) if den else np.nan

# ── Metric computation ────────────────────────────────────────────────────────
def compute_tm_pitch_metrics(df, cols, pitch_code):
    d = df[df['pitch_code']==pitch_code].copy()
    n = len(d)
    if n == 0: return {}

    # Swing/whiff from PitchCall
    pc_col = cols['pitch_call']
    if pc_col:
        pc = d[pc_col].astype(str).str.lower()
        swings = pc.str.contains('swing|foul|inplay|hit', regex=True)
        whiffs  = pc.str.contains('swingingstrike|strikeswinging|swinging_strike|missedswing|missed', regex=True)
        in_zone_pitches = None
        if cols['plate_x'] and cols['plate_z']:
            px = pd.to_numeric(d[cols['plate_x']], errors='coerce')
            pz = pd.to_numeric(d[cols['plate_z']], errors='coerce')
            in_zone = (px.abs() <= 17/24) & pz.between(1.5, 3.5)
            out_zone = ~in_zone
            chase = pc[out_zone].str.contains('swing|foul|inplay', regex=True)
            zone_pct  = _pct(in_zone.sum(), n)
            whiff_pct = _pct(whiffs.sum(), swings.sum())
            chase_pct = _pct(chase.sum(), out_zone.sum())
            swing_pct = _pct(swings.sum(), n)
        else:
            zone_pct=whiff_pct=chase_pct=swing_pct=np.nan
    else:
        zone_pct=whiff_pct=chase_pct=swing_pct=np.nan

    velo = pd.to_numeric(d[cols['velo']], errors='coerce').mean()  if cols['velo']  else np.nan
    spin = pd.to_numeric(d[cols['spin']], errors='coerce').mean()  if cols['spin']  else np.nan
    hb   = pd.to_numeric(d[cols['hb']],  errors='coerce').mean()  if cols['hb']    else np.nan
    ivb  = pd.to_numeric(d[cols['ivb']], errors='coerce').mean()  if cols['ivb']   else np.nan
    ext  = pd.to_numeric(d[cols['ext']], errors='coerce').mean()  if cols['ext']   else np.nan

    # TrackMan HB/IVB already in inches — no *12 needed
    return {'n':n,'velo':velo,'spin':spin,'hb':hb,'ivb':ivb,'ext':ext,
            'zone_pct':zone_pct,'whiff_pct':whiff_pct,'chase_pct':chase_pct,'swing_pct':swing_pct}

def compute_tm_batted_metrics(df, cols, pitch_code):
    d = df[df['pitch_code']==pitch_code].copy()
    pc_col = cols['pitch_call']
    ev_col = cols['exit_speed']
    la_col = cols['launch_angle']
    if pc_col:
        in_play = d[d[pc_col].astype(str).str.lower().str.contains('inplay|in_play|hit',regex=True)].copy()
    elif ev_col:
        in_play = d[pd.to_numeric(d[ev_col],errors='coerce').notna()].copy()
    else:
        return {}
    if ev_col: in_play = in_play[pd.to_numeric(in_play[ev_col],errors='coerce').notna()].copy()
    if in_play.empty: return {}
    n_bb = len(in_play)
    ev = pd.to_numeric(in_play[ev_col], errors='coerce') if ev_col else pd.Series([np.nan]*n_bb)
    la = pd.to_numeric(in_play[la_col], errors='coerce') if la_col else pd.Series([np.nan]*n_bb)
    avg_ev = float(ev.mean()) if ev_col else np.nan
    avg_la = float(la.mean()) if la_col else np.nan
    hh_mask = ev >= 95
    hh_pct  = _pct(hh_mask.sum(), n_bb)
    hh_la   = float(la[hh_mask].mean()) if hh_mask.any() else np.nan
    bm=(((ev>=98)&la.between(26,30))|((ev>=99)&la.between(25,31))|((ev>=100)&la.between(24,33))|
        ((ev>=101)&la.between(23,35))|((ev>=102)&la.between(22,38))|((ev>=103)&la.between(21,40))|
        ((ev>=104)&la.between(20,42))|((ev>=105)&la.between(19,44))|((ev>=106)&la.between(18,46))|
        ((ev>=107)&la.between(17,48))|((ev>=108)&la.between(16,50))|((ev>=109)&la.between(15,52))|
        ((ev>=110)&la.between(14,54)))
    barrel_pct = _pct(bm.sum(), n_bb)
    pr_col = cols['play_result']
    if pr_col:
        pr = in_play[pr_col].astype(str).str.lower()
        hits = pr.str.contains('single|double|triple|homerun|home_run',regex=True).sum()
        baa  = hits/n_bb if n_bb else np.nan
    else: baa=np.nan
    return {'avg_ev':avg_ev,'avg_la':avg_la,'hh_pct':hh_pct,'hh_la':hh_la,
            'barrel_pct':barrel_pct,'baa':baa,'n_bb':n_bb}

# ── Drawing helpers ───────────────────────────────────────────────────────────
def draw_strike_zone(ax, bottom=1.5, top=3.5):
    half=17/24
    ax.add_patch(Rectangle((-half-0.02,bottom-0.02),(half*2)+0.04,(top-bottom)+0.04,fill=False,edgecolor='#000000',linewidth=3.5,zorder=19,alpha=0.6))
    ax.add_patch(Rectangle((-half,bottom),half*2,top-bottom,fill=False,edgecolor='white',linewidth=1.8,zorder=20))
    for y in [bottom+(top-bottom)/3,bottom+2*(top-bottom)/3]:
        ax.plot([-half,half],[y,y],color='white',linewidth=0.5,alpha=0.35,zorder=21)
    for x in [-half/3,half/3]:
        ax.plot([x,x],[bottom,top],color='white',linewidth=0.5,alpha=0.35,zorder=21)

def draw_home_plate(ax, zone_bottom=1.5):
    hw=17/24; y0=zone_bottom-0.05; y1=zone_bottom-0.28
    ax.add_patch(Polygon([(-hw,y0),(hw,y0),(hw*0.55,y1),(0,y1-0.08),(-hw*0.55,y1)],
                          closed=True,facecolor='#444444',edgecolor='white',linewidth=1.0,zorder=20))

def create_heatmap(ax, x, y, xlim=(-2.0,2.0), ylim=(0.5,4.5)):
    ax.set_facecolor('#0a0a1a')
    x,y=np.asarray(x,float),np.asarray(y,float)
    mask=np.isfinite(x)&np.isfinite(y); x,y=x[mask],y[mask]
    if len(x)<5:
        ax.scatter(x,y,s=12,color='white',alpha=0.6,edgecolors='none',zorder=5); return
    xi=np.linspace(xlim[0],xlim[1],200); yi=np.linspace(ylim[0],ylim[1],200)
    xg,yg=np.meshgrid(xi,yi); pos=np.vstack([xg.ravel(),yg.ravel()])
    try:
        vals=np.vstack([x,y])
        if np.unique(x).size<2 or np.unique(y).size<2: raise ValueError
        kde=gaussian_kde(vals,bw_method='scott'); zi=kde(pos).reshape(xg.shape); zi=zi/zi.max()
        ax.contourf(xg,yg,zi,levels=30,cmap=SAVANT_CMAP,alpha=1.0,zorder=1)
        ax.contour(xg,yg,zi,levels=8,colors='k',linewidths=0.3,alpha=0.18,zorder=2)
    except Exception:
        ax.hexbin(x,y,gridsize=25,extent=(xlim[0],xlim[1],ylim[0],ylim[1]),cmap=SAVANT_CMAP,mincnt=1,zorder=1)

def draw_butterfly(ax, pitch_codes, df, cols):
    ax.set_facecolor(PANEL_COLOR)
    bs_col = cols['batter_side']
    lhh_pcts=[]; rhh_pcts=[]
    for p in pitch_codes:
        sub=df[df['pitch_code']==p]
        if bs_col:
            lhh=sub[sub[bs_col].astype(str).str.upper().isin(['L','LEFT'])]
            rhh=sub[sub[bs_col].astype(str).str.upper().isin(['R','RIGHT'])]
            tl=len(df[df[bs_col].astype(str).str.upper().isin(['L','LEFT'])]) or 1
            tr=len(df[df[bs_col].astype(str).str.upper().isin(['R','RIGHT'])]) or 1
        else:
            lhh=rhh=pd.DataFrame(); tl=tr=1
        lhh_pcts.append(len(lhh)/tl*100 if tl else 0)
        rhh_pcts.append(len(rhh)/tr*100 if tr else 0)
    n=len(pitch_codes); y_positions=list(range(n-1,-1,-1))
    axis_max=max(math.ceil(max(max(lhh_pcts),max(rhh_pcts),1)/25)*25,50)
    for i,(p,y) in enumerate(zip(pitch_codes,y_positions)):
        col=PITCH_COLORS.get(p,'#aaaaaa')
        ax.barh(y,-lhh_pcts[i],height=0.55,color=col,left=0,align='center',zorder=3)
        ax.barh(y, rhh_pcts[i],height=0.55,color=col,left=0,align='center',zorder=3)
        ax.text(-lhh_pcts[i]-1.2,y,f"{lhh_pcts[i]:.1f}%",ha='right',va='center',color=TEXT_COLOR,fontsize=13,fontweight='bold')
        ax.text( rhh_pcts[i]+1.2,y,f"{rhh_pcts[i]:.1f}%",ha='left', va='center',color=TEXT_COLOR,fontsize=13,fontweight='bold')
    ax.axvline(0,color='#cccccc',linewidth=1.2,zorder=5)
    for i,(p,y) in enumerate(zip(pitch_codes,y_positions)):
        ax.text(0,y+0.38,PITCH_NAMES.get(p,p),ha='center',va='bottom',
                color=PITCH_COLORS.get(p,'#aaaaaa'),fontsize=11,fontweight='bold',zorder=6)
    ax.set_xlim(-axis_max,axis_max); ax.set_ylim(-0.7,n-0.3); ax.set_yticks([])
    ticks=list(range(0,axis_max+1,25))
    ax.set_xticks([-t for t in ticks]+ticks)
    ax.set_xticklabels([f"{t}%" for t in ticks]*2,color=DIM_COLOR,fontsize=11)
    for t in range(25,axis_max+1,25):
        ax.axvline(t,color='#ffffff',linewidth=0.4,alpha=0.15,linestyle='--',zorder=1)
        ax.axvline(-t,color='#ffffff',linewidth=0.4,alpha=0.15,linestyle='--',zorder=1)
    ax.set_title('Pitch Usage',color=TEXT_COLOR,fontsize=14,fontweight='bold',pad=10,loc='center')
    ax.text(0.01,0.01,'vs LHH',transform=ax.transAxes,color=DIM_COLOR,fontsize=12,fontweight='bold',va='bottom',
            bbox=dict(boxstyle='round,pad=0.3',facecolor=PANEL_COLOR,edgecolor=BORDER_COLOR,linewidth=0.8))
    ax.text(0.99,0.01,'vs RHH',transform=ax.transAxes,color=DIM_COLOR,fontsize=12,fontweight='bold',va='bottom',ha='right',
            bbox=dict(boxstyle='round,pad=0.3',facecolor=PANEL_COLOR,edgecolor=BORDER_COLOR,linewidth=0.8))
    for spine in ax.spines.values(): spine.set_edgecolor(BORDER_COLOR)
    ax.tick_params(colors='#555555')

# ── Main card builder ─────────────────────────────────────────────────────────
def build_tm_card(df, pitcher_name, cols):
    hand_col = cols['throws']
    if hand_col and hand_col in df.columns:
        hand = df[hand_col].iloc[0]
        hand = 'R' if str(hand).upper() in ['R','RIGHT','RHP'] else 'L'
    else:
        hand = 'R'
    sign = -1 if hand=='R' else 1

    # Map pitch types
    df = df.copy()
    df['pitch_code'] = map_pitch_types(df, cols['pitch_type'])
    df = df[df['pitch_code'] != 'OTHER'].copy()

    counts = df['pitch_code'].value_counts()
    total  = counts.sum()
    pitch_codes = [p for p in counts.index if counts[p]/total >= 0.01]
    if not pitch_codes: raise ValueError("No valid pitch types found.")

    pitch_metrics  = {p: compute_tm_pitch_metrics(df, cols, p)  for p in pitch_codes}
    batted_metrics = {p: compute_tm_batted_metrics(df, cols, p) for p in pitch_codes}
    for p in pitch_codes: pitch_metrics[p]['usage_pct'] = pitch_metrics[p]['n']/total*100

    heat_pitch_cols=2; heat_rows=math.ceil(len(pitch_codes)/heat_pitch_cols)
    FIG_W=32; HEADER_H=3.0; BODY_H=9.0; HEAT_ROW_H=5.5; TABLE_H=2.8; BB_H=2.8; AGG_H=1.0
    fig_height=HEADER_H+BODY_H+heat_rows*HEAT_ROW_H+TABLE_H+BB_H+AGG_H
    fig=plt.figure(figsize=(FIG_W,fig_height),facecolor=BG_COLOR)
    outer=gridspec.GridSpec(6,1,figure=fig,
        height_ratios=[HEADER_H,BODY_H,heat_rows*HEAT_ROW_H,TABLE_H,BB_H,AGG_H],
        hspace=0.12,left=0.03,right=0.97,top=0.97,bottom=0.02)
    body_gs=gridspec.GridSpecFromSubplotSpec(1,2,subplot_spec=outer[1],width_ratios=[1.2,0.8],wspace=0.10)

    # HEADER
    ax_hdr=fig.add_subplot(outer[0]); ax_hdr.set_facecolor(PANEL_COLOR); ax_hdr.axis('off')
    ax_hdr.set_xlim(0,1); ax_hdr.set_ylim(0,1)
    ax_hdr.plot([0,1],[0.997,0.997],color='#cc0000',linewidth=4,transform=ax_hdr.transAxes,clip_on=False)
    throws_label='RHP' if hand=='R' else 'LHP'
    date_col=cols['game_date']
    if date_col and date_col in df.columns:
        dates=pd.to_datetime(df[date_col],errors='coerce').dropna()
        date_str=f"{dates.min().strftime('%m/%d/%y')} – {dates.max().strftime('%m/%d/%y')}" if len(dates) else ''
    else: date_str=''
    ax_hdr.text(0.01,0.72,pitcher_name.upper(),transform=ax_hdr.transAxes,color=TEXT_COLOR,
                fontsize=22,fontweight='bold',va='center',
                path_effects=[pe.withStroke(linewidth=2,foreground='black')])
    ax_hdr.text(0.01,0.22,f"{throws_label}  ·  {total} pitches  ·  TrackMan  {('· '+date_str) if date_str else ''}",
                transform=ax_hdr.transAxes,color=DIM_COLOR,fontsize=11,va='center')

    # Summary stat boxes
    all_pm = {p: pitch_metrics[p] for p in pitch_codes}
    fb_code = next((p for p in ['FF','SI','FC'] if p in pitch_codes), pitch_codes[0])
    fb_velo = pitch_metrics[fb_code].get('velo', np.nan)
    overall_whiff = _pct(sum(m.get('whiff_pct',0)*m.get('n',0)/100 for m in all_pm.values() if m.get('n',0)>0),
                         total) if total else np.nan

    def draw_stat_box(ax,x0,y0,w,h,stat_cols):
        ax.add_patch(FancyBboxPatch((x0,y0),w,h,boxstyle='round,pad=0.005',
            facecolor='#1a1a1a',edgecolor=BORDER_COLOR,linewidth=1.2,transform=ax.transAxes,zorder=2))
        n=len(stat_cols); cw=w/n
        for i,(lbl,val) in enumerate(stat_cols):
            cx=x0+cw*(i+0.5)
            ax.text(cx,y0+h*0.68,val,transform=ax.transAxes,color=TEXT_COLOR,fontsize=16,fontweight='bold',ha='center',va='center',zorder=3)
            ax.text(cx,y0+h*0.22,lbl,transform=ax.transAxes,color='#dddddd',fontsize=11,ha='center',va='center',zorder=3)
            if i<n-1: ax.plot([x0+cw*(i+1)]*2,[y0+0.04,y0+h-0.04],color=BORDER_COLOR,linewidth=0.7,transform=ax.transAxes,zorder=3)

    hdr_stats=[
        (PITCH_NAMES.get(fb_code,fb_code).split()[0]+' Velo', f"{fb_velo:.1f}" if np.isfinite(fb_velo) else '—'),
        ('Pitches', str(total)),
        ('Pitch Types', str(len(pitch_codes))),
        ('Overall Whiff%', f"{overall_whiff:.1f}%" if np.isfinite(overall_whiff) else '—'),
    ]
    draw_stat_box(ax_hdr,x0=0.30,y0=0.08,w=0.66,h=0.84,stat_cols=hdr_stats)

    # MOVEMENT PLOT
    ax_mov=fig.add_subplot(body_gs[0]); ax_mov.set_facecolor(PANEL_COLOR)
    ax_mov.grid(True,color='#ffffff',alpha=0.06,linewidth=0.7,zorder=0)
    ax_mov.axhline(0,color='#ffffff',alpha=0.15,linewidth=1.0,zorder=1)
    ax_mov.axvline(0,color='#ffffff',alpha=0.15,linewidth=1.0,zorder=1)
    lim=22; ax_mov.set_xlim(-lim,lim); ax_mov.set_ylim(-lim,lim); ax_mov.set_aspect('equal',adjustable='box')
    ax_mov.set_xlabel('← Glove Side          Arm Side →',color=DIM_COLOR,fontsize=13,labelpad=6)
    ax_mov.set_ylabel('← Drop          Rise →',color=DIM_COLOR,fontsize=13,labelpad=6)
    ax_mov.tick_params(colors='#555555',labelsize=9)
    for spine in ax_mov.spines.values(): spine.set_edgecolor(BORDER_COLOR)
    hb_col=cols['hb']; ivb_col=cols['ivb']
    for p in pitch_codes:
        col=PITCH_COLORS.get(p,'#aaaaaa'); sub=df[df['pitch_code']==p]
        if hb_col and ivb_col:
            hb_raw=pd.to_numeric(sub[hb_col],errors='coerce')*sign
            ivb_raw=pd.to_numeric(sub[ivb_col],errors='coerce')
            ax_mov.scatter(hb_raw,ivb_raw,s=12,color=col,alpha=0.55,edgecolors='none',zorder=3,rasterized=True)
    for p in pitch_codes:
        pm=pitch_metrics[p]; col=PITCH_COLORS.get(p,'#aaaaaa')
        hm=pm.get('hb',np.nan)*sign; vm=pm.get('ivb',np.nan)
        if np.isfinite(hm) and np.isfinite(vm):
            ax_mov.scatter(hm,vm,s=420,color=col,alpha=0.35,edgecolors='none',zorder=8)
            ax_mov.scatter(hm,vm,s=220,color=col,alpha=1.0,edgecolors='white',linewidths=1.4,zorder=9)
            ax_mov.text(hm,vm,PITCH_NAMES.get(p,p).split()[0][:3],color='white',fontsize=11,fontweight='bold',
                        ha='center',va='center',zorder=10,path_effects=[pe.withStroke(linewidth=1.5,foreground='black')])
    ax_mov.set_title('Movement Profile',color=TEXT_COLOR,fontsize=14,fontweight='bold',pad=10,loc='left')

    # BUTTERFLY
    draw_butterfly(fig.add_subplot(body_gs[1]),pitch_codes,df,cols)

    # PITCH METRICS TABLE
    col_labels=['Pitch','Use%','Velo','Spin','H-Break','IVB','Ext','Zone%','Swing%','Whiff%','Chase%']
    legend_rows=[]; row_colors=[]
    for p in pitch_codes:
        pm=pitch_metrics[p]; col=PITCH_COLORS.get(p,'#aaaaaa'); row_colors.append(col)
        def f1(v): return f"{v:.1f}" if np.isfinite(v) else "—"
        def f0(v): return f"{int(round(v))}" if np.isfinite(v) else "—"
        legend_rows.append([PITCH_NAMES.get(p,p),f"{pm.get('usage_pct',0):.1f}%",
            f1(pm.get('velo',np.nan)),f0(pm.get('spin',np.nan)),
            f"{pm.get('hb',np.nan)*sign:.1f}" if np.isfinite(pm.get('hb',np.nan)) else "—",
            f1(pm.get('ivb',np.nan)),f1(pm.get('ext',np.nan)),
            f"{pm.get('zone_pct',np.nan):.1f}%"  if np.isfinite(pm.get('zone_pct',np.nan))  else "—",
            f"{pm.get('swing_pct',np.nan):.1f}%" if np.isfinite(pm.get('swing_pct',np.nan)) else "—",
            f"{pm.get('whiff_pct',np.nan):.1f}%" if np.isfinite(pm.get('whiff_pct',np.nan)) else "—",
            f"{pm.get('chase_pct',np.nan):.1f}%" if np.isfinite(pm.get('chase_pct',np.nan)) else "—"])
    ax_leg=fig.add_subplot(outer[3]); ax_leg.set_facecolor(PANEL_COLOR); ax_leg.axis('off')
    n_rows=len(legend_rows); n_cols=len(col_labels); cell_h=1.0/(n_rows+1.5)
    for ci,lbl in enumerate(col_labels):
        ax_leg.text((ci+0.5)/n_cols,1-cell_h*0.6,lbl,ha='center',va='center',color='#dddddd',fontsize=14,fontweight='bold',transform=ax_leg.transAxes)
    for ri,(row,rcol) in enumerate(zip(legend_rows,row_colors)):
        y=1-cell_h*(ri+1.6)
        ax_leg.add_patch(FancyBboxPatch((0.001,y-cell_h*0.45),0.998,cell_h*0.9,boxstyle='round,pad=0.002',
            facecolor='#ffffff',alpha=0.07 if ri%2==0 else 0.03,edgecolor='none',transform=ax_leg.transAxes,zorder=0))
        ax_leg.add_patch(FancyBboxPatch((0.001,y-cell_h*0.3),0.003,cell_h*0.6,boxstyle='round,pad=0.001',
            facecolor=rcol,edgecolor='none',transform=ax_leg.transAxes,zorder=1))
        for ci,val in enumerate(row):
            ax_leg.text((ci+0.5)/n_cols,y,val,ha='center',va='center',
                        color=TEXT_COLOR if ci>0 else rcol,
                        fontsize=14,fontweight='bold' if ci==0 else 'normal',transform=ax_leg.transAxes)
    ax_leg.text(0.01,0.98,'PITCH METRICS',ha='left',va='top',color=TEXT_COLOR,fontsize=15,fontweight='bold',transform=ax_leg.transAxes)

    # HEATMAPS
    heat_gs=gridspec.GridSpecFromSubplotSpec(heat_rows,heat_pitch_cols*2,subplot_spec=outer[2],hspace=0.35,wspace=0.08)
    xlim_h=(-2.1,2.1); ylim_h=(0.4,4.6)
    px_col=cols['plate_x']; pz_col=cols['plate_z']; bs_col=cols['batter_side']
    for idx,p in enumerate(pitch_codes):
        row_i=idx//heat_pitch_cols; col_i=(idx%heat_pitch_cols)*2
        sub_all=df[df['pitch_code']==p]
        if px_col and pz_col:
            sub_all=sub_all.copy()
            sub_all['_px']=pd.to_numeric(sub_all[px_col],errors='coerce')
            sub_all['_pz']=pd.to_numeric(sub_all[pz_col],errors='coerce')
            sub_all=sub_all.dropna(subset=['_px','_pz'])
        col=PITCH_COLORS.get(p,'#aaaaaa'); pname=PITCH_NAMES.get(p,p)
        for h_idx,stand in enumerate(['L','R']):
            ax_h=fig.add_subplot(heat_gs[row_i,col_i+h_idx])
            if bs_col and px_col and pz_col:
                sub=sub_all[sub_all[bs_col].astype(str).str.upper().isin(['L','LEFT'] if stand=='L' else ['R','RIGHT'])]
                create_heatmap(ax_h,-sub['_px'].values,sub['_pz'].values,xlim=xlim_h,ylim=ylim_h)
                if len(sub)>=3:
                    ax_h.scatter(-sub['_px'].mean(),sub['_pz'].mean(),s=90,color=col,edgecolors='white',linewidths=1.4,zorder=25,alpha=0.95)
                n_sub=len(sub)
            else:
                ax_h.set_facecolor('#0a0a1a'); n_sub=0
            draw_strike_zone(ax_h); draw_home_plate(ax_h)
            ax_h.set_xlim(*xlim_h); ax_h.set_ylim(*ylim_h); ax_h.set_aspect('equal',adjustable='box'); ax_h.axis('off')
            ax_h.set_title(f"{pname}  {'vs LHH' if stand=='L' else 'vs RHH'}  (n={n_sub})",
                           color=TEXT_COLOR,fontsize=14,pad=5,fontweight='bold')
            if h_idx==0:
                ax_h.plot([xlim_h[0]+0.05]*2,[ylim_h[0],ylim_h[1]],color=col,linewidth=4,alpha=0.85,zorder=26)

    # BATTED BALL TABLE
    bb_labels=['Pitch','BIP','Avg EV','Hard Hit%','HH Launch Angle','Barrel%','BAA']
    bb_rows=[]
    for p in pitch_codes:
        bm=batted_metrics.get(p,{})
        def f1p(v): return f"{v:.1f}%" if np.isfinite(v) else "—"
        bb_rows.append([PITCH_NAMES.get(p,p),str(bm.get('n_bb','—')),
            f"{bm.get('avg_ev',np.nan):.1f} mph" if np.isfinite(bm.get('avg_ev',np.nan)) else "—",
            f1p(bm.get('hh_pct',np.nan)),
            f"{bm.get('hh_la',np.nan):.1f}°" if np.isfinite(bm.get('hh_la',np.nan)) else "—",
            f1p(bm.get('barrel_pct',np.nan)),
            f"{bm.get('baa',np.nan):.3f}" if np.isfinite(bm.get('baa',np.nan)) else "—"])
    ax_bb=fig.add_subplot(outer[4]); ax_bb.set_facecolor(PANEL_COLOR); ax_bb.axis('off')
    n_bb_cols=len(bb_labels); cell_h2=1.0/(len(bb_rows)+1.8)
    for ci,lbl in enumerate(bb_labels):
        ax_bb.text((ci+0.5)/n_bb_cols,1-cell_h2*0.65,lbl,ha='center',va='center',color='#dddddd',fontsize=14,fontweight='bold',transform=ax_bb.transAxes)
    ax_bb.plot([0.01,0.99],[1-cell_h2*1.05]*2,color=BORDER_COLOR,linewidth=0.8,transform=ax_bb.transAxes)
    for ri,(row,p) in enumerate(zip(bb_rows,pitch_codes)):
        y=1-cell_h2*(ri+1.65); rcol=PITCH_COLORS.get(p,'#aaaaaa')
        ax_bb.add_patch(FancyBboxPatch((0.001,y-cell_h2*0.45),0.998,cell_h2*0.9,boxstyle='round,pad=0.002',
            facecolor='#ffffff',alpha=0.07 if ri%2==0 else 0.03,edgecolor='none',transform=ax_bb.transAxes,zorder=0))
        ax_bb.add_patch(FancyBboxPatch((0.001,y-cell_h2*0.3),0.003,cell_h2*0.6,boxstyle='round,pad=0.001',
            facecolor=rcol,edgecolor='none',transform=ax_bb.transAxes,zorder=1))
        for ci,val in enumerate(row):
            ax_bb.text((ci+0.5)/n_bb_cols,y,val,ha='center',va='center',
                       color=rcol if ci==0 else TEXT_COLOR,
                       fontsize=14,fontweight='bold' if ci==0 else 'normal',transform=ax_bb.transAxes)
    ax_bb.text(0.01,0.98,'BATTED BALL OUTCOMES',ha='left',va='top',color=TEXT_COLOR,fontsize=15,fontweight='bold',transform=ax_bb.transAxes)

    # FOOTER
    ax_agg=fig.add_subplot(outer[5]); ax_agg.set_facecolor(PANEL_COLOR); ax_agg.axis('off')
    ax_agg.text(0.5,0.5,f"TrackMan Export  ·  {total} total pitches  ·  {len(pitch_codes)} pitch types",
                ha='center',va='center',color=DIM_COLOR,fontsize=11,transform=ax_agg.transAxes)
    fig.text(0.5,0.003,"Data: TrackMan  |  Pitch movement from pitcher's POV",
             ha='center',va='bottom',color='#555555',fontsize=9,style='italic')

    buf=BytesIO()
    plt.savefig(buf,dpi=180,bbox_inches='tight',facecolor=BG_COLOR,edgecolor='none',format='png')
    plt.close(fig); buf.seek(0)
    return buf

# ── Streamlit UI ──────────────────────────────────────────────────────────────
uploaded=st.file_uploader("Upload TrackMan CSV export",type="csv")

with st.expander("ℹ️ About TrackMan CSV format"):
    st.markdown("""
The app auto-detects column names from standard TrackMan exports. It looks for:
- **Pitcher** — pitcher name
- **PitcherThrows** — R or L
- **TaggedPitchType** or **AutoPitchType** — pitch classification
- **RelSpeed** — velocity
- **SpinRate, HorzBreak, InducedVertBreak, Extension** — movement metrics
- **PlateLocSide, PlateLocHeight** — location
- **BatterSide** — L or R
- **PitchCall** — result (StrikeSwinging, FoulBall, InPlay, etc.)
- **ExitSpeed, Angle** — batted ball data

If your columns are named differently, let me know and I can add them.
    """)

if uploaded:
    try:
        df_raw=pd.read_csv(uploaded)
        cols=detect_columns(df_raw)
        pitcher_col=cols['pitcher']

        if not pitcher_col:
            st.error("Could not find a Pitcher column. Check your CSV headers.")
        else:
            pitchers=sorted(df_raw[pitcher_col].dropna().unique().tolist())
            st.success(f"Loaded {len(df_raw):,} pitches · {len(pitchers)} pitchers")

            search=st.selectbox("Select pitcher",[""] + pitchers)
            if search:
                pitcher_df=df_raw[df_raw[pitcher_col]==search].copy()
                st.caption(f"{len(pitcher_df):,} pitches for {search}")
                if st.button("Generate Card",type="primary"):
                    with st.spinner("Building card..."):
                        buf=build_tm_card(pitcher_df,search,cols)
                    st.image(buf,use_column_width=True)
                    st.download_button("⬇ Download Card PNG",data=buf,
                                       file_name=f"{search.replace(' ','_')}_trackman.png",
                                       mime="image/png")
    except Exception as e:
        st.error(f"Error: {e}"); st.exception(e)
else:
    st.info("Upload a TrackMan CSV export above to get started.")
