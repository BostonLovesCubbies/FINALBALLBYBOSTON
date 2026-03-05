import streamlit as st
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyBboxPatch, Circle
from io import BytesIO
import warnings
warnings.filterwarnings("ignore")

st.set_page_config(page_title="MCAL Player Cards", page_icon="⚾", layout="wide")
st.title("MCAL Player Cards")

BG_COLOR='#0d0d0d'; PANEL_COLOR='#161616'; BORDER_COLOR='#333333'
TEXT_COLOR='#ffffff'; DIM_COLOR='#aaaaaa'

STAT_DEFS = [
    ('avg','AVG',True,'.3f'),('obp','OBP',True,'.3f'),('slg','SLG',True,'.3f'),
    ('ops','OPS',True,'.3f'),('woba','wOBA',True,'.3f'),('iso','ISO',True,'.3f'),
    ('hr','HR',True,'d'),('rbi','RBI',True,'d'),
    ('r','R',True,'d'),('h','H',True,'d'),('2b','2B',True,'d'),('3b','3B',True,'d'),
    ('sb','SB',True,'d'),('bb','BB',True,'d'),('k','K',False,'d'),
    ('kpct','K%',False,'.1f'),('bbpct','BB%',True,'.1f'),
]

DATA_PATH = 'hitters.csv'
LEADERBOARD_STATS = ['avg','obp','slg','ops','woba','iso','hr','rbi','sb']
LEADERBOARD_LABELS = {'avg':'AVG','obp':'OBP','slg':'SLG','ops':'OPS',
                      'woba':'wOBA','iso':'ISO','hr':'HR','rbi':'RBI','sb':'SB'}

def pct_color(pct):
    if pct >= 50:
        t=(pct-50)/50; r=int(255*t+220*(1-t)); g=int(50*t+220*(1-t)); b=int(50*t+220*(1-t))
    else:
        t=pct/50; r=int(220*t+40*(1-t)); g=int(220*t+80*(1-t)); b=int(220*t+255*(1-t))
    return f"#{r:02x}{g:02x}{b:02x}"

def get_percentile(value, series, higher_is_better=True):
    s=series.dropna()
    if len(s)==0: return 50
    return float(np.sum(s<=value)/len(s)*100) if higher_is_better else float(np.sum(s>=value)/len(s)*100)

def compute_stats(df):
    d=df.copy(); d.columns=[c.strip().lower() for c in d.columns]
    for col in ['hbp','sf','2b','3b','hr','bb','k','sb','r','rbi','g']:
        if col not in d.columns: d[col]=0
    d=d.fillna(0)
    for col in ['ab','h','2b','3b','hr','r','rbi','bb','k','sb','hbp','sf']:
        d[col]=pd.to_numeric(d[col],errors='coerce').fillna(0).astype(int)
    d['1b']=d['h']-d['2b']-d['3b']-d['hr']
    d['avg']=np.where(d['ab']>0,d['h']/d['ab'],0)
    obp_num=d['h']+d['bb']+d['hbp']; obp_den=d['ab']+d['bb']+d['hbp']+d['sf']
    d['obp']=np.where(obp_den>0,obp_num/obp_den,0)
    slg_num=d['1b']+2*d['2b']+3*d['3b']+4*d['hr']
    d['slg']=np.where(d['ab']>0,slg_num/d['ab'],0)
    d['ops']=d['obp']+d['slg']
    d['iso']=d['slg']-d['avg']
    pa=d['ab']+d['bb']+d['hbp']+d['sf']
    woba_num=(0.690*d['bb'])+(0.722*d['hbp'])+(0.881*d['1b'])+(1.254*d['2b'])+(1.594*d['3b'])+(2.058*d['hr'])
    woba_den=d['ab']+d['bb']+d['hbp']+d['sf']
    d['woba']=np.where(woba_den>0,woba_num/woba_den,0)
    d['kpct']=np.where(pa>0,d['k']/pa*100,0)
    d['bbpct']=np.where(pa>0,d['bb']/pa*100,0)
    d['pa']=pa
    d['player_norm']=d['player'].str.strip().str.lower()
    return d

@st.cache_data
def load_data():
    try:
        raw=pd.read_csv(DATA_PATH)
        return compute_stats(raw)
    except FileNotFoundError:
        return None

def fmt_val(val, fmt):
    if fmt=='.3f':
        return f"{val:.3f}".lstrip('0') if float(val)<1 else f"{val:.3f}"
    elif fmt=='.1f':
        return f"{val:.1f}%"
    else:
        return str(int(float(val)))

def build_card(p, qual, min_ab, selected_year, all_years):
    fig=plt.figure(figsize=(24,16),facecolor=BG_COLOR)
    outer=gridspec.GridSpec(3,1,figure=fig,height_ratios=[2.4,7.0,4.8],
                            hspace=0.18,left=0.03,right=0.97,top=0.97,bottom=0.04)
    pool_years=f"{min(all_years)}–{max(all_years)}" if len(all_years)>1 else str(all_years[0])
    is_qual=p['ab']>=min_ab

    # ── HEADER ────────────────────────────────────────────────────────────────
    ax_hdr=fig.add_subplot(outer[0])
    ax_hdr.set_facecolor(PANEL_COLOR); ax_hdr.axis('off')
    ax_hdr.set_xlim(0,1); ax_hdr.set_ylim(0,1)
    ax_hdr.plot([0,1],[0.998,0.998],color='#cc0000',linewidth=5,
                transform=ax_hdr.transAxes,clip_on=False)

    # Player name
    ax_hdr.text(0.01,0.76,p['player'].upper(),transform=ax_hdr.transAxes,
                color=TEXT_COLOR,fontsize=28,fontweight='bold',va='center')

    # Games box next to name
    g_val = int(float(p.get('g', 0)))
    ax_hdr.add_patch(FancyBboxPatch((0.215,0.58),0.065,0.36,boxstyle='round,pad=0.005',
        facecolor='#1c1c1c',edgecolor=BORDER_COLOR,linewidth=1.2,
        transform=ax_hdr.transAxes,zorder=2))
    ax_hdr.text(0.2475,0.82,'GAMES',transform=ax_hdr.transAxes,color=TEXT_COLOR,
                fontsize=8,fontweight='bold',ha='center',va='center',zorder=3)
    ax_hdr.text(0.2475,0.66,str(g_val),transform=ax_hdr.transAxes,color=TEXT_COLOR,
                fontsize=16,fontweight='bold',ha='center',va='center',zorder=3)
    # School / Pos / Year
    ax_hdr.text(0.01,0.30,f"{p['team']}  ·  {p['pos']}  ·  {selected_year}",
                transform=ax_hdr.transAxes,color=TEXT_COLOR,fontsize=16,
                fontweight='bold',va='center')
    # Qualified badge
    badge_col='#2ecc71' if is_qual else '#e74c3c'
    ax_hdr.text(0.01,0.03,
                f"{'QUALIFIED' if is_qual else 'NOT QUALIFIED'}  ({int(p['ab'])} AB)  "
                f"·  Percentiles vs {len(qual)}-player pool ({pool_years})",
                transform=ax_hdr.transAxes,color=badge_col,fontsize=9,
                fontweight='bold',va='bottom')

    # Stat boxes
    keys=['avg','obp','slg','ops','woba','iso','hr','rbi','sb','ab']
    labels={'avg':'AVG','obp':'OBP','slg':'SLG','ops':'OPS','woba':'wOBA',
            'iso':'ISO','hr':'HR','rbi':'RBI','sb':'SB','ab':'AB'}
    n_boxes=len(keys); bx0=0.28; bw=0.70; bh=0.92; by0=0.04; cw=bw/n_boxes
    ax_hdr.add_patch(FancyBboxPatch((bx0,by0),bw,bh,boxstyle='round,pad=0.005',
        facecolor='#1c1c1c',edgecolor=BORDER_COLOR,linewidth=1.2,
        transform=ax_hdr.transAxes,zorder=2))
    for i,key in enumerate(keys):
        cx=bx0+cw*(i+0.5); val=p.get(key,0)
        fmt=next((s[3] for s in STAT_DEFS if s[0]==key),'d')
        val_str=fmt_val(val,fmt)
        sd=next((s for s in STAT_DEFS if s[0]==key),None)
        col=(pct_color(get_percentile(float(val),qual[key],sd[2]))
             if sd and key in qual.columns and len(qual)>0 else TEXT_COLOR)
        ax_hdr.text(cx,by0+bh*0.75,labels[key],transform=ax_hdr.transAxes,
                    color=TEXT_COLOR,fontsize=10,fontweight='bold',
                    ha='center',va='center',zorder=3)
        ax_hdr.text(cx,by0+bh*0.28,val_str,transform=ax_hdr.transAxes,color=col,
                    fontsize=15,fontweight='bold',ha='center',va='center',zorder=3)
        if i<n_boxes-1:
            ax_hdr.plot([bx0+cw*(i+1)]*2,[by0+0.05,by0+bh-0.05],
                        color=BORDER_COLOR,linewidth=0.7,
                        transform=ax_hdr.transAxes,zorder=3)

    # ── PERCENTILE SECTION ────────────────────────────────────────────────────
    def draw_pct_section(ax, stat_keys, section_label):
        ax.set_facecolor(PANEL_COLOR); ax.axis('off')
        ax.set_xlim(0,1); ax.set_ylim(0,1)
        n=len(stat_keys)
        row_h=0.86/n
        y_start=0.91

        # Section header
        ax.text(0.01,0.98,section_label,ha='left',va='top',color=TEXT_COLOR,
                fontsize=13,fontweight='bold',transform=ax.transAxes)
        ax.text(0.99,0.98,f'{pool_years} league pool',ha='right',va='top',
                color=TEXT_COLOR,fontsize=10,fontweight='bold',transform=ax.transAxes)
        ax.plot([0.01,0.99],[0.94,0.94],color=BORDER_COLOR,linewidth=0.6,
                transform=ax.transAxes)

        # Column headers
        ax.text(0.01,0.925,'STAT',ha='left',va='top',color=DIM_COLOR,
                fontsize=8,transform=ax.transAxes)
        ax.text(0.22,0.925,'VALUE',ha='left',va='top',color=DIM_COLOR,
                fontsize=8,transform=ax.transAxes)
        ax.text(0.99,0.925,'PCT',ha='right',va='top',color=DIM_COLOR,
                fontsize=8,transform=ax.transAxes)

        for i,key in enumerate(stat_keys):
            sd=next((s for s in STAT_DEFS if s[0]==key),None)
            if not sd: continue
            _,label,higher,fmt=sd
            val=float(p.get(key,0))
            pct=get_percentile(val,qual[key],higher) if key in qual.columns and len(qual)>0 else 50
            color=pct_color(pct)
            y=y_start-row_h*(i+0.5)

            # Stat label — white, bold, left
            ax.text(0.01,y,label,ha='left',va='center',color=TEXT_COLOR,
                    fontsize=13,fontweight='bold',transform=ax.transAxes)

            # Stat value — white, next to label
            val_str=fmt_val(val,fmt)
            ax.text(0.22,y,val_str,ha='left',va='center',color=TEXT_COLOR,
                    fontsize=13,fontweight='bold',transform=ax.transAxes)

            # Bar track — slim like Savant
            bl=0.38; br=0.82
            bw2=br-bl; bh2=row_h*0.18; by2=y-bh2/2
            ax.add_patch(FancyBboxPatch((bl,by2),bw2,bh2,
                boxstyle='round,pad=0.001',facecolor='#2a2a2a',
                edgecolor='none',transform=ax.transAxes,zorder=1))

            # Fill bar
            fill_w=max(bw2*pct/100,0.003)
            ax.add_patch(FancyBboxPatch((bl,by2),fill_w,bh2,
                boxstyle='round,pad=0.001',facecolor=color,
                edgecolor='none',transform=ax.transAxes,zorder=2))

            # 50th percentile tick
            ax.plot([bl+bw2*0.5]*2,[by2-0.005,by2+bh2+0.005],
                    color='#555555',linewidth=1.0,transform=ax.transAxes,zorder=3)

            # Savant-style circle — perfectly round using scatter
            circle_x=0.915
            ax.scatter([circle_x],[y],s=(row_h*420)**2,color=color,
                      edgecolors='white',linewidths=1.5,zorder=5,
                      transform=ax.transAxes,clip_on=False)
            ax.text(circle_x,y,str(int(pct)),ha='center',va='center',
                    color='white',fontsize=10,fontweight='bold',
                    transform=ax.transAxes,zorder=6)

    body_gs=gridspec.GridSpecFromSubplotSpec(1,2,subplot_spec=outer[1],wspace=0.08)
    draw_pct_section(fig.add_subplot(body_gs[0]),
                     ['avg','obp','slg','ops','woba','iso','kpct','bbpct'],
                     'RATE STATS')
    draw_pct_section(fig.add_subplot(body_gs[1]),
                     ['hr','rbi','r','h','2b','3b','sb','bb','k'],
                     'COUNTING STATS')

    # ── COUNTING TABLE ────────────────────────────────────────────────────────
    ax_tbl=fig.add_subplot(outer[2])
    ax_tbl.set_facecolor(PANEL_COLOR); ax_tbl.axis('off')
    ax_tbl.set_xlim(0,1); ax_tbl.set_ylim(0,1)
    tbl_cols=['ab','h','1b','2b','3b','hr','r','rbi','bb','k','sb','hbp','pa']
    tbl_labels=['AB','H','1B','2B','3B','HR','R','RBI','BB','K','SB','HBP','PA']
    cw3=1.0/len(tbl_cols)

    # Section title
    ax_tbl.text(0.01,0.97,f'{selected_year} SEASON STATS',ha='left',va='top',
                color=TEXT_COLOR,fontsize=13,fontweight='bold',
                transform=ax_tbl.transAxes)
    ax_tbl.text(0.99,0.97,
                f"League pool: {len(qual)} qualified players  ·  {pool_years}  ·  min {min_ab} AB",
                ha='right',va='top',color=DIM_COLOR,fontsize=9,
                transform=ax_tbl.transAxes)
    ax_tbl.plot([0.01,0.99],[0.88,0.88],color=BORDER_COLOR,linewidth=0.6,
                transform=ax_tbl.transAxes)

    # Column headers
    for ci,lbl in enumerate(tbl_labels):
        ax_tbl.text((ci+0.5)*cw3,0.78,lbl,ha='center',va='center',
                    color=TEXT_COLOR,fontsize=13,fontweight='bold',
                    transform=ax_tbl.transAxes)
    ax_tbl.plot([0.01,0.99],[0.65,0.65],color=BORDER_COLOR,linewidth=0.6,
                transform=ax_tbl.transAxes)

    # Player values
    for ci,key in enumerate(tbl_cols):
        val=p.get(key,0)
        sd=next((s for s in STAT_DEFS if s[0]==key),None)
        col=(pct_color(get_percentile(float(val),qual[key],sd[2]))
             if sd and key in qual.columns and len(qual)>0 else TEXT_COLOR)
        ax_tbl.text((ci+0.5)*cw3,0.44,str(int(float(val))),ha='center',va='center',
                    color=col,fontsize=16,fontweight='bold',transform=ax_tbl.transAxes)

    # League avg row
    ax_tbl.plot([0.01,0.99],[0.28,0.28],color=BORDER_COLOR,linewidth=0.5,
                transform=ax_tbl.transAxes)
    ax_tbl.text(0.002,0.16,'LG AVG',ha='left',va='center',color=DIM_COLOR,
                fontsize=8,fontweight='bold',transform=ax_tbl.transAxes)
    for ci,key in enumerate(tbl_cols):
        if key in qual.columns:
            avg_val=qual[key].mean()
            ax_tbl.text((ci+0.5)*cw3,0.16,f"{avg_val:.1f}",ha='center',va='center',
                        color=TEXT_COLOR,fontsize=10,fontweight='bold',
                        transform=ax_tbl.transAxes)

    fig.text(0.5,0.032,
             'Stats via MaxPreps  ·  wOBA uses 2025 MLB linear weights  ·  K% and BB% per PA  ·  Percentiles vs multi-year qualified league pool  ·  Cards created by Boston Nahigian',
             ha='center',va='bottom',color='#888888',fontsize=8,fontweight='bold')

    buf=BytesIO()
    plt.savefig(buf,dpi=220,bbox_inches='tight',facecolor=BG_COLOR,
                edgecolor='none',format='png')
    plt.close(fig); buf.seek(0)
    return buf

# ── Load data ─────────────────────────────────────────────────────────────────
df=load_data()
if df is None:
    st.error("No data file found. Make sure `hitters.csv` is in the repo root.")
    st.stop()

all_years=sorted(df['year'].astype(int).unique().tolist())
min_ab=int(df['ab'].quantile(0.40))
qual=df[df['ab']>=min_ab].copy()
pool_years=f"{min(all_years)}–{max(all_years)}" if len(all_years)>1 else str(all_years[0])

# ── LEADERBOARD ───────────────────────────────────────────────────────────────
st.markdown("### 🏆 Leaderboard")
lb_col1,lb_col2=st.columns(2)
with lb_col1:
    lb_stat=st.selectbox("Sort by",LEADERBOARD_STATS,
                         format_func=lambda x: LEADERBOARD_LABELS[x])
with lb_col2:
    lb_year=st.selectbox("Season",
                         ["All years"]+[str(y) for y in sorted(all_years,reverse=True)])

lb_df=qual.copy()
if lb_year != "All years":
    lb_df=lb_df[lb_df['year'].astype(int)==int(lb_year)]

if not lb_df.empty and lb_stat in lb_df.columns:
    top=lb_df.nlargest(10,lb_stat)[['player','team','year',lb_stat,'ab']].copy()
    top['year']=top['year'].astype(int)
    fmt=next((s[3] for s in STAT_DEFS if s[0]==lb_stat),'.3f')
    top[lb_stat]=top[lb_stat].apply(lambda x: fmt_val(x,fmt))
    top.columns=['Player','Team','Year',LEADERBOARD_LABELS[lb_stat],'AB']
    top.index=range(1,len(top)+1)
    st.dataframe(top,use_container_width=True)

st.markdown("---")

# ── PLAYER SEARCH ─────────────────────────────────────────────────────────────
st.markdown("### 🔍 Player Card")
col1,col2=st.columns([2,1])
with col1:
    search=st.selectbox("Search player",[""] + sorted(df['player'].unique().tolist()))
with col2:
    if search:
        player_years=sorted(
            df[df['player_norm']==search.strip().lower()]['year'].astype(int).unique().tolist(),
            reverse=True)
        selected_year=st.selectbox("Season",
                                   ["Select a year..."]+[str(y) for y in player_years])
    else:
        st.selectbox("Season",["—"],disabled=True)
        selected_year="Select a year..."

if search and selected_year not in ["Select a year...","—",""]:
    season_row=df[
        (df['player_norm']==search.strip().lower()) &
        (df['year'].astype(int)==int(selected_year))
    ]
    if season_row.empty:
        st.warning("No data found for that player/season.")
    else:
        p=season_row.iloc[0]
        with st.spinner("Building card..."):
            buf=build_card(p,qual,min_ab,selected_year,all_years)
        st.image(buf,use_container_width=True)
        st.download_button("⬇ Download Card PNG",data=buf,
                           file_name=f"{search.replace(' ','_')}_{selected_year}_card.png",
                           mime="image/png")
