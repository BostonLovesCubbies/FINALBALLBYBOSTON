import math
import re
import warnings
import numpy as np
import pandas as pd
import requests
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patheffects as pe
from matplotlib.patches import Rectangle, Polygon, FancyBboxPatch
from matplotlib.colors import LinearSegmentedColormap
from scipy.stats import gaussian_kde
from io import BytesIO

warnings.filterwarnings("ignore")

try:
    import pybaseball as pyb
    pyb.cache.enable()
except ImportError:
    raise ImportError("Run: pip install pybaseball")

PITCH_COLORS = {
    'FF': '#D22D49', 'SI': '#C57A02', 'FC': '#933F2C', 'SL': '#DDB33A',
    'ST': '#A2C8FC', 'CU': '#00D1ED', 'KC': '#3BACAC', 'CH': '#1DBE3A',
    'FS': '#3CB84B', 'SV': '#9B59B6', 'KN': '#888888', 'EP': '#aaaaaa',
    'FO': '#55aa55', 'SC': '#cc8844', 'CS': '#00aadd', 'PO': '#999999',
    'IN': '#bbbbbb', 'AB': '#cccccc',
}
PITCH_NAMES = {
    'FF': '4-Seam Fastball', 'SI': 'Sinker',     'FC': 'Cutter',
    'SL': 'Slider',          'ST': 'Sweeper',     'CU': 'Curveball',
    'KC': 'Knuckle Curve',   'CH': 'Changeup',    'FS': 'Splitter',
    'SV': 'Slurve',          'KN': 'Knuckleball', 'EP': 'Eephus',
    'FO': 'Forkball',        'SC': 'Screwball',   'CS': 'Slow Curve',
}
HEATMAP_COLORS = ['#0a0a2e','#1a1a6e','#2244aa','#4477cc','#77aaee',
                  '#aaccff','#ffffff','#ffcc66','#ff8833','#ee3311','#aa0000']
SAVANT_CMAP  = LinearSegmentedColormap.from_list('savant', HEATMAP_COLORS, N=256)
BG_COLOR     = '#0d0d0d'
PANEL_COLOR  = '#161616'
BORDER_COLOR = '#2a2a2a'
TEXT_COLOR   = '#e8e8e8'
DIM_COLOR    = '#aaaaaa'

SWING_RE = re.compile(
    r'swinging_strike_blocked|swinging_strike|foul_tip|foul_bunt|missed_bunt'
    r'|hit_into_play|(?<!pitchout_)foul(?!_pitchout)', re.I)
WHIFF_RE = re.compile(r'swinging_strike_blocked|swinging_strike|missed_bunt', re.I)
CONTACT_EVENTS = {'single','double','triple','home_run','field_out','force_out',
                  'grounded_into_double_play','sac_fly','sac_bunt','field_error','double_play'}
HIT_EVENTS = {'single','double','triple','home_run'}
NON_AB     = {'walk','hit_by_pitch','sac_bunt','sac_fly','intent_walk'}

def _pct(num, den): return (num/den*100) if den else np.nan

def fetch_pitcher_data(pitcher_id: int, year: int = 2025):
    start = f"{year}-03-20"
    end   = f"{year}-10-31"
    df = pyb.statcast_pitcher(start, end, player_id=pitcher_id)
    if df is None or df.empty:
        raise ValueError(f"No {year} Statcast data for pitcher {pitcher_id}.")
    return df, year

def fetch_player_info(pitcher_id: int):
    url = f"https://statsapi.mlb.com/api/v1/people/{pitcher_id}"
    try:
        r = requests.get(url, timeout=6); r.raise_for_status()
        person = r.json().get('people',[{}])[0]
        team   = person.get('currentTeam',{})
        return {'name': person.get('fullName', f'Pitcher {pitcher_id}'),
                'team_abbr': team.get('abbreviation',''),
                'team_id':   team.get('id'),
                'throws':    person.get('pitchHand',{}).get('code','R')}
    except Exception:
        return {'name': f'Pitcher {pitcher_id}','team_abbr':'','team_id':None,'throws':'R'}

def fetch_season_stats(pitcher_id: int, year: int):
    url = (f"https://statsapi.mlb.com/api/v1/people/{pitcher_id}/stats"
           f"?stats=season&season={year}&group=pitching")
    out = {'IP':'—','ERA':'—','xERA':'—','K-BB%':'—'}
    try:
        r = requests.get(url, timeout=6); r.raise_for_status()
        splits = r.json().get('stats',[{}])[0].get('splits',[{}])
        if not splits: return out
        stat = splits[0].get('stat',{})
        ip=stat.get('inningsPitched'); era=stat.get('era')
        k=stat.get('strikeOuts'); bb=stat.get('baseOnBalls'); bf=stat.get('battersFaced')
        out['IP']  = str(ip) if ip else '—'
        out['ERA'] = f"{float(era):.2f}" if era else '—'
        if k is not None and bb is not None and bf and float(bf)>0:
            out['K-BB%'] = f"{(float(k)-float(bb))/float(bf)*100:.1f}%"
    except Exception: pass
    return out

def fetch_team_logo(team_id, timeout=6):
    if not team_id: return None
    try:
        from PIL import Image
        resp = requests.get(f"https://www.mlbstatic.com/team-logos/{team_id}.png", timeout=timeout)
        resp.raise_for_status()
        return Image.open(BytesIO(resp.content)).convert("RGBA")
    except Exception: return None

def fetch_fangraphs_stats(player_name: str, year: int):
    out = {k:'—' for k in ['FIP','xFIP','SIERA','K%','BB%','K-BB%','WAR','GB%','FB%','LD%']}
    try:
        df = pyb.pitching_stats(year, year, qual=1)
        if df is None or df.empty: return out
        def norm(s): return str(s).strip().lower()
        target = norm(player_name)
        name_col = next((c for c in ['Name','name','PlayerName'] if c in df.columns), None)
        if not name_col: return out
        row = df[df[name_col].apply(norm)==target]
        if row.empty:
            last = target.split()[-1]
            row = df[df[name_col].apply(lambda x: norm(x).split()[-1])==last]
        if row.empty: return out
        row = row.iloc[0]
        def col(candidates):
            for c in candidates:
                if c in row.index and pd.notna(row[c]): return row[c]
            return None
        def pct(v):
            if v is None: return '—'
            v=float(v); v=v*100 if v<1.0 else v; return f"{v:.1f}%"
        def dec(v, fmt='.2f'): return f"{float(v):{fmt}}" if v is not None else '—'
        fip=col(['FIP']); xfip=col(['xFIP']); siera=col(['SIERA'])
        kpct=col(['K%']); bbpct=col(['BB%']); war=col(['WAR'])
        gb=col(['GB%']); fb=col(['FB%']); ld=col(['LD%'])
        k_val  = float(kpct) *(1 if kpct  is not None and float(kpct) >1 else 100) if kpct  is not None else None
        bb_val = float(bbpct)*(1 if bbpct is not None and float(bbpct)>1 else 100) if bbpct is not None else None
        kmbb = f"{k_val-bb_val:.1f}%" if k_val is not None and bb_val is not None else '—'
        out.update({'FIP':dec(fip),'xFIP':dec(xfip),'SIERA':dec(siera),
                    'K%':pct(kpct),'BB%':pct(bbpct),'K-BB%':kmbb,
                    'WAR':dec(war,'.1f'),'GB%':pct(gb),'FB%':pct(fb),'LD%':pct(ld)})
    except Exception: pass
    return out

def compute_pitch_metrics(df, pitch_type=None):
    d = df[df['pitch_type']==pitch_type].copy() if pitch_type else df.copy()
    d = d.dropna(subset=['description']); n=len(d)
    if n==0: return {}
    desc=d['description'].astype(str)
    swings=desc.str.contains(SWING_RE); whiffs=desc.str.contains(WHIFF_RE)
    if 'plate_x' in d.columns and 'plate_z' in d.columns:
        in_zone=((d['plate_x'].abs()<=17/24)&
                 d['plate_z'].between(d.get('sz_bot',pd.Series([1.5]*len(d))).fillna(1.5),
                                      d.get('sz_top',pd.Series([3.5]*len(d))).fillna(3.5)))
        out_zone=~in_zone; chase_swings=desc[out_zone].str.contains(SWING_RE)
        zone_pct=_pct(in_zone.sum(),n); whiff_pct=_pct(whiffs.sum(),swings.sum())
        chase_pct=_pct(chase_swings.sum(),out_zone.sum())
    else: zone_pct=whiff_pct=chase_pct=np.nan
    velo=d['release_speed'].mean()     if 'release_speed'     in d else np.nan
    spin=d['release_spin_rate'].mean() if 'release_spin_rate' in d else np.nan
    hmov=d['pfx_x'].mean()*12         if 'pfx_x'             in d else np.nan
    vmov=d['pfx_z'].mean()*12         if 'pfx_z'             in d else np.nan
    ext =d['release_extension'].mean() if 'release_extension' in d else np.nan
    return {'n':n,'velo':velo,'spin':spin,'hmov':hmov,'vmov':vmov,'ext':ext,
            'zone_pct':zone_pct,'whiff_pct':whiff_pct,'chase_pct':chase_pct,
            'swing_pct':_pct(swings.sum(),n)}

def compute_batted_ball_metrics(df, pitch_type=None):
    d=df[df['pitch_type']==pitch_type].copy() if pitch_type else df.copy()
    if 'type' in d.columns:     bb=d[d['type']=='X'].copy()
    elif 'events' in d.columns: bb=d[d['events'].isin(CONTACT_EVENTS)].copy()
    else:                        bb=pd.DataFrame()
    if 'launch_speed' in bb.columns: bb=bb[bb['launch_speed'].notna()].copy()
    if bb.empty: return {}
    n_bb=len(bb)
    avg_ev=float(bb['launch_speed'].mean()) if 'launch_speed' in bb else np.nan
    avg_la=float(bb['launch_angle'].mean()) if 'launch_angle' in bb else np.nan
    hh_mask=bb['launch_speed']>=95 if 'launch_speed' in bb.columns else pd.Series([],dtype=bool)
    hh_pct=_pct(hh_mask.sum(),n_bb)
    hh_la=float(bb.loc[hh_mask,'launch_angle'].mean()) if 'launch_angle' in bb.columns and hh_mask.any() else np.nan
    if 'barrel' in bb.columns: barrel_pct=_pct(bb['barrel'].fillna(0).sum(),n_bb)
    elif 'launch_speed' in bb.columns and 'launch_angle' in bb.columns:
        ev=bb['launch_speed']; la=bb['launch_angle']
        bm=(((ev>=98)&la.between(26,30))|((ev>=99)&la.between(25,31))|((ev>=100)&la.between(24,33))|
            ((ev>=101)&la.between(23,35))|((ev>=102)&la.between(22,38))|((ev>=103)&la.between(21,40))|
            ((ev>=104)&la.between(20,42))|((ev>=105)&la.between(19,44))|((ev>=106)&la.between(18,46))|
            ((ev>=107)&la.between(17,48))|((ev>=108)&la.between(16,50))|((ev>=109)&la.between(15,52))|
            ((ev>=110)&la.between(14,54)))
        barrel_pct=_pct(bm.sum(),n_bb)
    else: barrel_pct=np.nan
    xw_col=next((c for c in ['estimated_woba_using_speedangle','estimated_woba','xwoba'] if c in bb.columns),None)
    xs_col=next((c for c in ['estimated_slugging_using_speedangle','estimated_slg','xslg'] if c in bb.columns),None)
    xwoba=float(bb[xw_col].dropna().mean()) if xw_col else np.nan
    xslg =float(bb[xs_col].dropna().mean()) if xs_col else np.nan
    if 'events' in bb.columns:
        hits=bb['events'].isin(HIT_EVENTS).sum()
        abs_=(~bb['events'].isin(NON_AB)&bb['events'].notna()).sum()
        baa=hits/abs_ if abs_ else np.nan
    else: baa=np.nan
    return {'avg_ev':avg_ev,'avg_la':avg_la,'hh_pct':hh_pct,'hh_la':hh_la,
            'barrel_pct':barrel_pct,'xwoba':xwoba,'xslg':xslg,'baa':baa,'n_bb':n_bb}

def draw_strike_zone(ax, bottom=1.5, top=3.5):
    half=17/24
    ax.add_patch(Rectangle((-half-0.02,bottom-0.02),(half*2)+0.04,(top-bottom)+0.04,
                            fill=False,edgecolor='#000000',linewidth=3.5,zorder=19,alpha=0.6))
    ax.add_patch(Rectangle((-half,bottom),half*2,top-bottom,
                            fill=False,edgecolor='white',linewidth=1.8,zorder=20))
    for y in [bottom+(top-bottom)/3,bottom+2*(top-bottom)/3]:
        ax.plot([-half,half],[y,y],color='white',linewidth=0.5,alpha=0.35,zorder=21)
    for x in [-half/3,half/3]:
        ax.plot([x,x],[bottom,top],color='white',linewidth=0.5,alpha=0.35,zorder=21)

def draw_home_plate(ax, zone_bottom=1.5):
    hw=17/24; y0=zone_bottom-0.05; y1=zone_bottom-0.28
    ax.add_patch(Polygon([(-hw,y0),(hw,y0),(hw*0.55,y1),(0,y1-0.08),(-hw*0.55,y1)],
                          closed=True,facecolor='#444444',edgecolor='white',linewidth=1.0,zorder=20))

def create_heatmap(ax, x, y, xlim=(-2.0,2.0), ylim=(0.5,4.5), cmap=None):
    if cmap is None: cmap=SAVANT_CMAP
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
        ax.contourf(xg,yg,zi,levels=30,cmap=cmap,alpha=1.0,zorder=1)
        ax.contour(xg,yg,zi,levels=8,colors='k',linewidths=0.3,alpha=0.18,zorder=2)
    except Exception:
        ax.hexbin(x,y,gridsize=25,extent=(xlim[0],xlim[1],ylim[0],ylim[1]),cmap=cmap,mincnt=1,zorder=1)

def draw_butterfly_usage(ax, pitch_types, df, pitch_colors, pitch_names):
    ax.set_facecolor(PANEL_COLOR)
    lhh_pcts=[]; rhh_pcts=[]
    for p in pitch_types:
        sub=df[df['pitch_type']==p]
        lhh=sub[sub['stand']=='L'] if 'stand' in sub.columns else pd.DataFrame()
        rhh=sub[sub['stand']=='R'] if 'stand' in sub.columns else pd.DataFrame()
        total_l=len(df[df['stand']=='L']) if 'stand' in df.columns else 1
        total_r=len(df[df['stand']=='R']) if 'stand' in df.columns else 1
        lhh_pcts.append(len(lhh)/total_l*100 if total_l else 0)
        rhh_pcts.append(len(rhh)/total_r*100 if total_r else 0)
    n=len(pitch_types); y_positions=list(range(n-1,-1,-1))
    max_pct=max(max(lhh_pcts),max(rhh_pcts),1)
    axis_max=max(math.ceil(max_pct/25)*25,50)
    for i,(p,y) in enumerate(zip(pitch_types,y_positions)):
        col=pitch_colors.get(p,'#aaaaaa')
        ax.barh(y,-lhh_pcts[i],height=0.55,color=col,left=0,align='center',zorder=3)
        ax.barh(y, rhh_pcts[i],height=0.55,color=col,left=0,align='center',zorder=3)
        ax.text(-lhh_pcts[i]-1.2,y,f"{lhh_pcts[i]:.1f}%",ha='right',va='center',color=TEXT_COLOR,fontsize=13,fontweight='bold')
        ax.text( rhh_pcts[i]+1.2,y,f"{rhh_pcts[i]:.1f}%",ha='left', va='center',color=TEXT_COLOR,fontsize=13,fontweight='bold')
    ax.axvline(0,color='#cccccc',linewidth=1.2,zorder=5)
    for i,(p,y) in enumerate(zip(pitch_types,y_positions)):
        ax.text(0,y+0.38,pitch_names.get(p,p),ha='center',va='bottom',
                color=PITCH_COLORS.get(p,'#aaaaaa'),fontsize=11,fontweight='bold',zorder=6)
    ax.set_xlim(-axis_max,axis_max); ax.set_ylim(-0.7,n-0.3); ax.set_yticks([])
    ticks=list(range(0,axis_max+1,25))
    ax.set_xticks([-t for t in ticks]+ticks)
    ax.set_xticklabels([f"{t}%" for t in ticks]+[f"{t}%" for t in ticks],color=DIM_COLOR,fontsize=11)
    for t in range(25,axis_max+1,25):
        ax.axvline( t,color='#ffffff',linewidth=0.4,alpha=0.15,linestyle='--',zorder=1)
        ax.axvline(-t,color='#ffffff',linewidth=0.4,alpha=0.15,linestyle='--',zorder=1)
    ax.set_title('Pitch Usage',color=TEXT_COLOR,fontsize=14,fontweight='bold',pad=10,loc='center')
    ax.text(0.01,0.01,'vs LHH',transform=ax.transAxes,color=DIM_COLOR,fontsize=12,fontweight='bold',va='bottom',
            bbox=dict(boxstyle='round,pad=0.3',facecolor=PANEL_COLOR,edgecolor=BORDER_COLOR,linewidth=0.8))
    ax.text(0.99,0.01,'vs RHH',transform=ax.transAxes,color=DIM_COLOR,fontsize=12,fontweight='bold',va='bottom',ha='right',
            bbox=dict(boxstyle='round,pad=0.3',facecolor=PANEL_COLOR,edgecolor=BORDER_COLOR,linewidth=0.8))
    for spine in ax.spines.values(): spine.set_edgecolor(BORDER_COLOR)
    ax.tick_params(colors='#555555')

def build_chart(pitcher_id: int, year: int = 2025):
    df, year    = fetch_pitcher_data(pitcher_id, year)
    info        = fetch_player_info(pitcher_id)
    season_stat = fetch_season_stats(pitcher_id, year)
    fg_stat     = fetch_fangraphs_stats(info['name'], year)
    logo        = fetch_team_logo(info['team_id'])
    hand        = df['p_throws'].iloc[0] if 'p_throws' in df.columns else info['throws']
    sign        = -1 if hand=='R' else 1
    valid_pitches = df[df['pitch_type'].notna()&(df['pitch_type']!='')].copy()
    counts=valid_pitches['pitch_type'].value_counts(); total_pitches=counts.sum()
    pitch_types=[p for p in counts.index if counts[p]/total_pitches>=0.01]
    if not pitch_types: raise ValueError("No valid pitch types found.")
    pitch_metrics ={p:compute_pitch_metrics(valid_pitches,p) for p in pitch_types}
    batted_metrics={p:compute_batted_ball_metrics(df,p)      for p in pitch_types}
    for p in pitch_types: pitch_metrics[p]['usage_pct']=pitch_metrics[p]['n']/total_pitches*100

    heat_pitch_cols=2; heat_rows=math.ceil(len(pitch_types)/heat_pitch_cols)
    FIG_W=32; HEADER_H=3.6; BODY_H=9.0; HEAT_ROW_H=5.5; TABLE_H=2.8; BB_H=2.8; AGG_H=1.8
    fig_height=HEADER_H+BODY_H+heat_rows*HEAT_ROW_H+TABLE_H+BB_H+AGG_H
    fig=plt.figure(figsize=(FIG_W,fig_height),facecolor=BG_COLOR)
    outer=gridspec.GridSpec(6,1,figure=fig,
        height_ratios=[HEADER_H,BODY_H,heat_rows*HEAT_ROW_H,TABLE_H,BB_H,AGG_H],
        hspace=0.12,left=0.03,right=0.97,top=0.97,bottom=0.02)
    body_gs=gridspec.GridSpecFromSubplotSpec(1,2,subplot_spec=outer[1],width_ratios=[1.2,0.8],wspace=0.10)

    # HEADER
    ax_hdr=fig.add_subplot(outer[0]); ax_hdr.set_facecolor(PANEL_COLOR); ax_hdr.axis('off')
    ax_hdr.set_xlim(0,1); ax_hdr.set_ylim(0,1)
    ax_hdr.plot([0,1],[0.995,0.995],color='#cc0000',linewidth=4,transform=ax_hdr.transAxes,clip_on=False)
    throws_label='RHP' if hand=='R' else 'LHP'
    ax_hdr.text(0.01,0.75,info['name'].upper(),transform=ax_hdr.transAxes,color=TEXT_COLOR,
                fontsize=24,fontweight='bold',va='center',
                path_effects=[pe.withStroke(linewidth=2,foreground='black')])
    ax_hdr.text(0.01,0.25,f"{info['team_abbr']}  ·  {throws_label}  ·  {year}",
                transform=ax_hdr.transAxes,color=DIM_COLOR,fontsize=12,va='center')
    def draw_stat_box(ax,x0,y0,w,h,cols):
        ax.add_patch(FancyBboxPatch((x0,y0),w,h,boxstyle='round,pad=0.005',
            facecolor='#1a1a1a',edgecolor=BORDER_COLOR,linewidth=1.2,transform=ax.transAxes,zorder=2))
        n=len(cols); col_w=w/n
        for i,(lbl,val) in enumerate(cols):
            cx=x0+col_w*(i+0.5)
            ax.text(cx,y0+h*0.68,val,transform=ax.transAxes,color=TEXT_COLOR,fontsize=16,fontweight='bold',ha='center',va='center',zorder=3)
            ax.text(cx,y0+h*0.22,lbl,transform=ax.transAxes,color='#dddddd',fontsize=11,ha='center',va='center',zorder=3)
            if i<n-1:
                ax.plot([x0+col_w*(i+1)]*2,[y0+0.04,y0+h-0.04],color=BORDER_COLOR,linewidth=0.7,transform=ax.transAxes,zorder=3)
    draw_stat_box(ax_hdr,x0=0.30,y0=0.08,w=0.66,h=0.84,cols=[
        ('IP',season_stat.get('IP','—')),('ERA',season_stat.get('ERA','—')),
        ('xERA',season_stat.get('xERA','—')),('FIP',fg_stat.get('FIP','—')),
        ('xFIP',fg_stat.get('xFIP','—')),('fWAR',fg_stat.get('WAR','—')),
        ('K-BB%',season_stat.get('K-BB%','—'))])
    if logo is not None:
        from matplotlib.offsetbox import OffsetImage, AnnotationBbox
        ax_hdr.add_artist(AnnotationBbox(OffsetImage(np.array(logo),zoom=0.50,alpha=0.9),
                          (0.975,0.5),xycoords='axes fraction',frameon=False))

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
    for p in pitch_types:
        col=PITCH_COLORS.get(p,'#aaaaaa'); sub=valid_pitches[valid_pitches['pitch_type']==p]
        ax_mov.scatter(sub['pfx_x']*12*sign,sub['pfx_z']*12,s=12,color=col,alpha=0.55,edgecolors='none',zorder=3,rasterized=True)
    for p in pitch_types:
        pm=pitch_metrics[p]; col=PITCH_COLORS.get(p,'#aaaaaa')
        hm=pm.get('hmov',np.nan)*sign; vm=pm.get('vmov',np.nan)
        if np.isfinite(hm) and np.isfinite(vm):
            ax_mov.scatter(hm,vm,s=420,color=col,alpha=0.35,edgecolors='none',zorder=8)
            ax_mov.scatter(hm,vm,s=220,color=col,alpha=1.0,edgecolors='white',linewidths=1.4,zorder=9)
            ax_mov.text(hm,vm,PITCH_NAMES.get(p,p).split()[0][:3],color='white',fontsize=11,fontweight='bold',
                        ha='center',va='center',zorder=10,path_effects=[pe.withStroke(linewidth=1.5,foreground='black')])
    ax_mov.set_title('Movement Profile',color=TEXT_COLOR,fontsize=14,fontweight='bold',pad=10,loc='left')

    # BUTTERFLY
    draw_butterfly_usage(fig.add_subplot(body_gs[1]),pitch_types,valid_pitches,PITCH_COLORS,PITCH_NAMES)

    # PITCH METRICS TABLE
    col_labels=['Pitch','Use%','Velo','Spin','H-Mov','V-Mov','Ext','Zone%','Swing%','Whiff%','Chase%']
    legend_rows=[]; row_colors=[]
    for p in pitch_types:
        pm=pitch_metrics[p]; col=PITCH_COLORS.get(p,'#aaaaaa'); row_colors.append(col)
        def f1(v): return f"{v:.1f}" if np.isfinite(v) else "—"
        def f0(v): return f"{int(round(v))}" if np.isfinite(v) else "—"
        legend_rows.append([PITCH_NAMES.get(p,p),f"{pm.get('usage_pct',np.nan):.1f}%",
            f1(pm.get('velo',np.nan)),f0(pm.get('spin',np.nan)),
            f"{pm.get('hmov',np.nan)*sign:.1f}" if np.isfinite(pm.get('hmov',np.nan)) else "—",
            f1(pm.get('vmov',np.nan)),f1(pm.get('ext',np.nan)),
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
    for idx,p in enumerate(pitch_types):
        row_i=idx//heat_pitch_cols; col_i=(idx%heat_pitch_cols)*2
        sub_all=valid_pitches[valid_pitches['pitch_type']==p].dropna(subset=['plate_x','plate_z'])
        col=PITCH_COLORS.get(p,'#aaaaaa'); pname=PITCH_NAMES.get(p,p)
        for h_idx,stand in enumerate(['L','R']):
            ax_h=fig.add_subplot(heat_gs[row_i,col_i+h_idx])
            sub=sub_all[sub_all['stand']==stand] if 'stand' in sub_all.columns else sub_all
            create_heatmap(ax_h,-sub['plate_x'].values,sub['plate_z'].values,xlim=xlim_h,ylim=ylim_h)
            draw_strike_zone(ax_h); draw_home_plate(ax_h)
            if len(sub)>=3:
                ax_h.scatter(-sub['plate_x'].mean(),sub['plate_z'].mean(),s=90,color=col,
                             edgecolors='white',linewidths=1.4,zorder=25,alpha=0.95)
            ax_h.set_xlim(*xlim_h); ax_h.set_ylim(*ylim_h); ax_h.set_aspect('equal',adjustable='box'); ax_h.axis('off')
            ax_h.set_title(f"{pname}  {'vs LHH' if stand=='L' else 'vs RHH'}  (n={len(sub)})",
                           color=TEXT_COLOR,fontsize=14,pad=5,fontweight='bold')
            if h_idx==0:
                ax_h.plot([xlim_h[0]+0.05]*2,[ylim_h[0],ylim_h[1]],color=col,linewidth=4,alpha=0.85,zorder=26)

    # BATTED BALL TABLE
    bb_labels=['Pitch','BIP','Avg EV','Hard Hit%','HH Launch Angle','Barrel%','xwOBAcon','xSLGcon','BAA']
    bb_rows=[]
    for p in pitch_types:
        bm=batted_metrics.get(p,{})
        def f3(v): return f"{v:.3f}" if np.isfinite(v) else "—"
        def f1p(v): return f"{v:.1f}%" if np.isfinite(v) else "—"
        bb_rows.append([PITCH_NAMES.get(p,p),str(bm.get('n_bb','—')),
            f"{bm.get('avg_ev',np.nan):.1f} mph" if np.isfinite(bm.get('avg_ev',np.nan)) else "—",
            f1p(bm.get('hh_pct',np.nan)),
            f"{bm.get('hh_la',np.nan):.1f}°" if np.isfinite(bm.get('hh_la',np.nan)) else "—",
            f1p(bm.get('barrel_pct',np.nan)),f3(bm.get('xwoba',np.nan)),
            f3(bm.get('xslg',np.nan)),f3(bm.get('baa',np.nan))])
    ax_bb=fig.add_subplot(outer[4]); ax_bb.set_facecolor(PANEL_COLOR); ax_bb.axis('off')
    n_bb_cols=len(bb_labels); n_bb_rows=len(bb_rows); cell_h2=1.0/(n_bb_rows+1.8)
    for ci,lbl in enumerate(bb_labels):
        ax_bb.text((ci+0.5)/n_bb_cols,1-cell_h2*0.65,lbl,ha='center',va='center',color='#dddddd',fontsize=14,fontweight='bold',transform=ax_bb.transAxes)
    ax_bb.plot([0.01,0.99],[1-cell_h2*1.05]*2,color=BORDER_COLOR,linewidth=0.8,transform=ax_bb.transAxes,clip_on=True)
    for ri,(row,p) in enumerate(zip(bb_rows,pitch_types)):
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

    # AGGREGATE STATS
    ax_agg=fig.add_subplot(outer[5]); ax_agg.set_facecolor(PANEL_COLOR); ax_agg.axis('off')
    agg_labels=['K%','BB%','K-BB%','SIERA','GB%','FB%','LD%','fWAR']
    agg_vals=[fg_stat.get(k,'—') for k in ['K%','BB%','K-BB%','SIERA','GB%','FB%','LD%','WAR']]
    n_agg=len(agg_labels); cell_w=1.0/n_agg
    for ci,lbl in enumerate(agg_labels):
        ax_agg.text((ci+0.5)*cell_w,0.82,lbl,ha='center',va='center',color='#dddddd',fontsize=13,fontweight='bold',transform=ax_agg.transAxes)
    ax_agg.plot([0.01,0.99],[0.62,0.62],color=BORDER_COLOR,linewidth=0.8,transform=ax_agg.transAxes)
    for ci,val in enumerate(agg_vals):
        ax_agg.text((ci+0.5)*cell_w,0.30,val,ha='center',va='center',color=TEXT_COLOR,fontsize=14,fontweight='bold',transform=ax_agg.transAxes)
        if ci<n_agg-1:
            ax_agg.plot([(ci+1)*cell_w]*2,[0.05,0.95],color=BORDER_COLOR,linewidth=0.5,transform=ax_agg.transAxes,alpha=0.4)
    ax_agg.add_patch(FancyBboxPatch((0.005,0.03),0.99,0.94,boxstyle='round,pad=0.005',
        facecolor='#1a1a1a',edgecolor=BORDER_COLOR,linewidth=1.0,transform=ax_agg.transAxes,zorder=0))
    ax_agg.text(0.01,0.97,'AGGREGATE STATS',ha='left',va='top',color=TEXT_COLOR,fontsize=12,fontweight='bold',transform=ax_agg.transAxes)
    fig.text(0.5,0.005,"Data: Baseball Savant  ·  FanGraphs  |  Pitch movement from pitcher's POV  |  @ballbyboston",
             ha='center',va='bottom',color='#888888',fontsize=10,style='italic')

    # ── Return PNG bytes instead of saving to disk ────────────────────────────
    buf = BytesIO()
    plt.savefig(buf, dpi=200, bbox_inches='tight', facecolor=BG_COLOR, edgecolor='none', format='png')
    plt.close(fig)
    buf.seek(0)
    return buf, info['name'], year
