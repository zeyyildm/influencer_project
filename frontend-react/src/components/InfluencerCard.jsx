import React from 'react'

/* ── helpers ──────────────────────────────────────────────── */
const rankCls = r => r===1?'rk1':r===2?'rk2':r===3?'rk3':'rkn'

const catCls = (c='') => ({
  moda:'bc-moda', spor:'bc-spor', lifestyle:'bc-lifestyle',
  teknoloji:'bc-tekno', yemek:'bc-yemek', 'anne-bebek':'bc-saglik',
  saglik:'bc-saglik', eglence:'bc-eglence', oyun:'bc-oyun',
  seyahat:'bc-seyahat', egitim:'bc-egitim', evcilhayvan:'bc-other',
  'diğer':'bc-other',
}[c.toLowerCase()] ?? 'bc-other')

const accCls = a => ({'mega':'bc-mega','makro':'bc-makro','mikro':'bc-mikro',
  'creator':'bc-makro','business':'bc-mega','personal':'bc-mikro'}[a] ?? 'bc-mikro')
const mlCls  = m => ({'uygun':'ml-u','orta':'ml-o','uygun_degil':'ml-x','uygunsuz':'ml-x'}[m] ?? 'ml-o')
const mlText = m => ({
  uygun: 'Uygun',
  orta: 'Orta',
  uygun_degil: 'Uygun Değil',
  uygunsuz: 'Uygun Değil',
  bilinmiyor: 'Model Yok',
}[m] ?? 'Model Yok')
const mlExplain = m => ({
  uygun: 'XGBoost modeli kampanya, kategori ve skor featurelarını birlikte olumlu değerlendirdi.',
  orta: 'Model sinyali kararsız; insan onayı veya ek marka kriteri önerilir.',
  uygun_degil: 'Model feature kombinasyonunu riskli/eşleşme dışı olarak işaretledi; sıralama puanı düşürüldü.',
  uygunsuz: 'Model feature kombinasyonunu riskli/eşleşme dışı olarak işaretledi; sıralama puanı düşürüldü.',
  bilinmiyor: 'Model dosyası veya feature şeması hazır olmadığı için tahmin üretilmedi.',
}[m] ?? 'Model dosyası veya feature şeması hazır olmadığı için tahmin üretilmedi.')

const sourceCls  = s => s === 'instagram' ? 'src-instagram' : s === 'synthetic' ? 'src-synthetic' : 'src-unknown'
const sourceText = s => s === 'instagram' ? 'Instagram' : s === 'synthetic' ? 'Sentetik' : 'Kaynak Yok'

const fitLevel = score => {
  if (score >= 75) return 'Çok güçlü'
  if (score >= 60) return 'Güçlü'
  if (score >= 45) return 'Orta'
  return 'Zayıf'
}

const riskLevel = risk => {
  if (risk <= 15) return 'Düşük risk'
  if (risk <= 40) return 'Orta risk'
  return 'Yüksek risk'
}

const recommendationReason = ({ ml_label, sfs, NFS, cfs, data_source }) => {
  const source = data_source === 'instagram' ? 'gerçek Instagram verisiyle' : 'sentetik veri kaynağıyla'
  if (ml_label === 'uygun_degil' || ml_label === 'uygunsuz') {
    return `İçerik sinyali ${fitLevel(sfs).toLowerCase()} olsa da AI uygunluk modeli bu profili riskli gördü; son skor bu nedenle düşürüldü.`
  }
  if (ml_label === 'orta') {
    return `Bu profil ${source} eşleşti; kampanya uyumu ${fitLevel(cfs).toLowerCase()}, karar için ek insan kontrolü önerilir.`
  }
  return `Bu profil ${source} eşleşti; içerik uyumu ${fitLevel(sfs).toLowerCase()} ve performans sinyali ${fitLevel(NFS).toLowerCase()}.`
}

const fmt = n => {
  if (n == null || n === 0) return null
  if (n >= 1_000_000) return `${(n/1_000_000).toFixed(1)}M`
  if (n >= 1_000)     return `${(n/1_000).toFixed(0)}K`
  return String(Math.round(n))
}

const riskColor = v => v<=15?'#10b981':v<=40?'#f59e0b':'#ef4444'
const riskBg    = v => v<=15?'rgba(16,185,129,0.1)':v<=40?'rgba(245,158,11,0.1)':'rgba(239,68,68,0.1)'

const stripeGradient = (score, rank) => {
  if (rank===1) return 'linear-gradient(90deg,#fbbf24,#f97316,#fb923c)'
  if (rank===2) return 'linear-gradient(90deg,#94a3b8,#64748b,#475569)'
  if (rank===3) return 'linear-gradient(90deg,#f97316,#ef4444,#fb7185)'
  if (score>=70) return 'linear-gradient(90deg,#34d399,#06b6d4,#38bdf8)'
  if (score>=50) return 'linear-gradient(90deg,#7c72f8,#8b5cf6,#38bdf8)'
  return 'linear-gradient(90deg,#a78bfa,#7c72f8,#6366f1)'
}

const arcColors = score =>
  score>=70 ? ['#10b981','#06b6d4'] :
  score>=50 ? ['#5046e5','#8b5cf6'] :
              ['#f59e0b','#fb7185']

const SCORE_COLORS = { NFS:'#5046e5', SFS:'#0891b2', CFS:'#7c3aed', BAS:'#059669' }
const SCORE_INFO = {
  SFS: { title: 'İçerik Uyumu',   desc: 'Marka metniyle paylaşım dilinin anlamsal yakınlığı' },
  NFS: { title: 'Performans',     desc: 'Etkileşim oranı ve paylaşım ritmi sinyali' },
  CFS: { title: 'Kampanya Uyumu', desc: 'Bu kampanya tipine benzer içeriklerdeki uygunluk' },
  BAS: { title: 'Marka Uyumu',    desc: 'Semantik uyum, kalite ve risk sinyallerinin özeti' },
}

/* ── icons ───────────────────────────────────────────────── */
const UsersIcon = () => (
  <svg viewBox="0 0 16 16" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round">
    <circle cx="6" cy="5" r="2.5"/>
    <path d="M1.5 13.5c0-2.5 2-4.5 4.5-4.5s4.5 2 4.5 4.5"/>
    <path d="M11 7c1.4 0 2.5 1.1 2.5 2.5 0 1-.6 1.9-1.5 2.3" opacity=".5"/>
    <path d="M13 13.5c0-1-.3-2-.9-2.7" opacity=".5"/>
  </svg>
)
const ActivityIcon = () => (
  <svg viewBox="0 0 16 16" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round">
    <polyline points="1,9 4,9 6,3 8,13 10,6 12,9 15,9"/>
  </svg>
)
const TrendIcon = () => (
  <svg viewBox="0 0 16 16" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round">
    <polyline points="1,12 5.5,7 8.5,9.5 15,2"/>
    <polyline points="11,2 15,2 15,6"/>
  </svg>
)
const ShieldIcon = () => (
  <svg viewBox="0 0 16 16" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round">
    <path d="M8 2L3 4.5v3.5C3 11 8 14 8 14s5-3 5-6V4.5L8 2z"/>
    <path d="M5.5 8l2 2 3-3"/>
  </svg>
)

/* ── arc gauge ───────────────────────────────────────────── */
function ScoreArc({ value, rank }) {
  const r=26, cx=32, cy=32
  const circ = 2*Math.PI*r
  const pct  = Math.min(value??0, 100)/100
  const id   = `ag${rank}`
  const [c1,c2] = arcColors(value??0)
  return (
    <div className="score-arc-wrap">
      <svg width="64" height="64" viewBox="0 0 64 64">
        <defs>
          <linearGradient id={id} x1="1" y1="0" x2="0" y2="1">
            <stop offset="0%" stopColor={c1}/>
            <stop offset="100%" stopColor={c2}/>
          </linearGradient>
        </defs>
        <circle cx={cx} cy={cy} r={r} fill="none"
          stroke="rgba(80,70,229,0.1)" strokeWidth="5"/>
        <circle cx={cx} cy={cy} r={r} fill="none"
          stroke={`url(#${id})`} strokeWidth="5"
          strokeDasharray={circ}
          strokeDashoffset={circ*(1-pct)}
          strokeLinecap="round"
          transform={`rotate(-90 ${cx} ${cy})`}
          style={{transition:'stroke-dashoffset 1.2s cubic-bezier(.4,0,.2,1)'}}
        />
      </svg>
      <div className="arc-overlay">
        <span className="arc-num">{value!=null?value.toFixed(1):'—'}</span>
        <span className="arc-lbl">Final</span>
      </div>
    </div>
  )
}

/* ── score item ──────────────────────────────────────────── */
function ScoreItem({ label, value, cls }) {
  const pct  = value!=null ? Math.min(value,100) : 0
  const info = SCORE_INFO[label] ?? { title: label, desc: '' }
  return (
    <div className="score-item" title={`${label}: ${info.desc}`}>
      <div className="score-row-label">
        <span className="sc-label-wrap">
          <span className="sc-title">{info.title}</span>
          <span className="sc-code" style={{color:SCORE_COLORS[label]}}>{label}</span>
        </span>
        <span className="sc-val">{value!=null?value.toFixed(1):'—'}</span>
      </div>
      <div className="sc-desc">{info.desc}</div>
      <div className="sc-track">
        <div className={`sc-fill ${cls}`} style={{width:`${pct}%`}}/>
      </div>
    </div>
  )
}

/* ── stat cell ───────────────────────────────────────────── */
function StatCell({ Icon, label, value, iconBg, iconColor, valueColor }) {
  return (
    <div className="stat-cell">
      <div className="sc-icon-box" style={{background:iconBg, color:iconColor}}>
        <Icon/>
      </div>
      <span className="sc-v" style={valueColor?{color:valueColor}:undefined}>{value}</span>
      <span className="sc-lbl">{label}</span>
    </div>
  )
}

/* ── main card ───────────────────────────────────────────── */
export default function InfluencerCard({ rank, data }) {
  const {
    influencer_name, category, account_type,
    NFS, sfs, cfs, campaign_bas,
    raw_final_score, ai_adjustment, final_score,
    positive_ratio, fake_followers_risk,
    avg_views, engagement_rate,
    ml_label, data_source,
  } = data

  const risk         = fake_followers_risk ?? 0
  const hasAiAdj     = ai_adjustment != null && Number(ai_adjustment) !== 0
  const rawScore     = raw_final_score ?? final_score
  const reason       = recommendationReason({ ml_label, sfs, NFS, cfs, data_source })

  return (
    <div className={`icard${rank===1?' top1':''}${mlCls(ml_label)==='ml-x'?' ai-risk':''}`}>
      <div className="icard-stripe" style={{background:stripeGradient(final_score,rank)}}/>
      <div className="icard-body">

        {/* header */}
        <div className="icard-head">
          <div className="icard-left">
            <div className={`rank ${rankCls(rank)}`}>#{rank}</div>
            <div className="icard-info">
              <div className="icard-name" title={influencer_name}>{influencer_name}</div>
              <div className="tag-row">
                {category     && <span className={`badge ${catCls(category)}`}>{category}</span>}
                {account_type && <span className={`badge ${accCls(account_type)}`}>{account_type}</span>}
                {data_source  && <span className={`badge ${sourceCls(data_source)}`}>{sourceText(data_source)}</span>}
              </div>
            </div>
          </div>
          <ScoreArc value={final_score} rank={rank}/>
        </div>

        <div className="icard-div"/>

        {/* decision strip */}
        <div className="decision-strip">
          <div>
            <span className="decision-kicker">Genel uygunluk</span>
            <strong>{fitLevel(final_score)}</strong>
          </div>
          <div>
            <span className="decision-kicker">Risk</span>
            <strong className={risk>40?'risk-high':risk>15?'risk-mid':'risk-low'}>
              {riskLevel(risk)}
            </strong>
          </div>
        </div>

        {/* reason */}
        <div className="reason-box">
          <span>Neden önerildi?</span>
          <p>{reason}</p>
        </div>

        {/* AI panel */}
        <div className={`xai-panel ${mlCls(ml_label)}`}>
          <div className="xai-top">
            <span className="xai-title">AI Uygunluk Analizi Durumu</span>
            <span className="xai-badge">{mlText(ml_label)}</span>
          </div>
          <p>{mlExplain(ml_label)}</p>
          <div className="xai-score-line">
            <span>Ham Final: {rawScore!=null?rawScore.toFixed(1):'---'}</span>
            <span className={hasAiAdj?'xai-penalty':'xai-neutral'}>
              AI Etkisi: {ai_adjustment!=null?Number(ai_adjustment).toFixed(1):'0.0'}
            </span>
          </div>
        </div>

        {/* score grid */}
        <div className="score-grid">
          <ScoreItem label="SFS" value={sfs}          cls="f-sfs"/>
          <ScoreItem label="NFS" value={NFS}          cls="f-nfs"/>
          <ScoreItem label="CFS" value={cfs}          cls="f-cfs"/>
          <ScoreItem label="BAS" value={campaign_bas} cls="f-bas"/>
        </div>

        {/* stats row — avg_views ve engagement_rate kullanıyor */}
        <div className="stats-row">
          <StatCell
            Icon={UsersIcon}
            label="Ort. İzlenme"
            value={fmt(avg_views) ?? '—'}
            iconBg="rgba(80,70,229,0.1)"
            iconColor="#5046e5"
          />
          <StatCell
            Icon={ActivityIcon}
            label="Etkileşim"
            value={engagement_rate!=null?`%${Number(engagement_rate).toFixed(1)}`:'—'}
            iconBg="rgba(8,145,178,0.1)"
            iconColor="#0891b2"
            valueColor="#0891b2"
          />
          <StatCell
            Icon={TrendIcon}
            label="Pozitif"
            value={positive_ratio!=null?`%${Number(positive_ratio).toFixed(0)}`:'—'}
            iconBg="rgba(16,185,129,0.1)"
            iconColor="#10b981"
            valueColor="#10b981"
          />
          <StatCell
            Icon={ShieldIcon}
            label="Sahte Risk"
            value={`%${risk.toFixed(0)}`}
            iconBg={riskBg(risk)}
            iconColor={riskColor(risk)}
            valueColor={riskColor(risk)}
          />
        </div>

      </div>
    </div>
  )
}
