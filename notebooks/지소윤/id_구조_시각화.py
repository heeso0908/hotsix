import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import platform

if platform.system() == 'Windows':
    plt.rcParams['font.family'] = 'Malgun Gothic'
elif platform.system() == 'Darwin':
    plt.rcParams['font.family'] = 'AppleGothic'
else:
    plt.rcParams['font.family'] = 'NanumGothic'
plt.rcParams['axes.unicode_minus'] = False

t1 = pd.read_csv('../../data/df_final_Type1.csv')
t2 = pd.read_csv('../../data/df_final_Type2.csv')

for df in [t1, t2]:
    df['id_str']  = df['id'].astype(str).str.zfill(7)
    df['생산코드'] = df['id_str'].str[:1]
    df['순번']    = df['id_str'].str[:4].astype(int)

t1['타입'] = 'Type1'
t2['타입'] = 'Type2'
df_all = pd.concat([t1, t2], ignore_index=True)

colors = {
    '0': '#E74C3C', '1': '#E67E22', '2': '#F39C12',
    '3': '#2ECC71', '4': '#1ABC9C', '5': '#3498DB',
    '6': '#9B59B6', '7': '#E91E63'
}

fig = plt.figure(figsize=(16, 14), facecolor='#F8FAFB')
fig.suptitle('id 구조 분석', fontsize=15, fontweight='bold', y=0.99)

# -------------------------------------------------------
# 1. id 구조 설명 텍스트
# -------------------------------------------------------
ax0 = fig.add_subplot(4, 1, 1)
ax0.axis('off')
ax0.set_facecolor('#EBF5FB')

ax0.text(0.5, 0.80, 'id 구조:  [ 앞 4자리: 생산 순번 ] + [ 뒤 3자리: Shot 번호 ]',
         ha='center', fontsize=13, fontweight='bold', transform=ax0.transAxes)
ax0.text(0.5, 0.50,
         '예)  id = 0,001,002  →  순번: 0001 (생산코드 0)  /  Shot: 002',
         ha='center', fontsize=11, color='#555555', transform=ax0.transAxes)
ax0.text(0.5, 0.20,
         'Type1: 생산코드 0~4  (순번 0000~4205)          Type2: 생산코드 4~7  (순번 4207~7533)',
         ha='center', fontsize=11, color='#2C3E50', transform=ax0.transAxes)

# -------------------------------------------------------
# 2. 생산코드별 순번 범위 (간트 차트)
# -------------------------------------------------------
ax1 = fig.add_subplot(4, 1, 2)
grp = df_all.groupby('생산코드')['순번'].agg(['min', 'max'])
for idx, (code, row) in enumerate(grp.iterrows()):
    color = colors.get(code, 'gray')
    ax1.barh(idx, row['max'] - row['min'], left=row['min'],
             color=color, alpha=0.85, height=0.6)
    label = "코드 {} ({:,}~{:,})".format(code, int(row['min']), int(row['max']))
    ax1.text((row['min'] + row['max']) / 2, idx, label,
             ha='center', va='center', fontsize=9, fontweight='bold', color='white')

ax1.set_yticks(range(len(grp)))
ax1.set_yticklabels(['코드 {}'.format(c) for c in grp.index])
ax1.set_xlabel('순번 (id 앞 4자리)')
ax1.set_title('생산코드별 순번 범위', fontweight='bold')
ax1.axvline(4206, color='black', linestyle='--', lw=1.5, label='Type1/Type2 경계')
ax1.legend(loc='upper right')
ax1.grid(axis='x', linestyle='--', alpha=0.4)

# -------------------------------------------------------
# 3. 순번 × Shot 번호 스캐터
# -------------------------------------------------------
ax2 = fig.add_subplot(4, 1, 3)
for code in sorted(df_all['생산코드'].unique()):
    sub = df_all[df_all['생산코드'] == code]
    ax2.scatter(sub['순번'], sub['Shot'], s=4, alpha=0.4,
                color=colors.get(code, 'gray'), label='코드 {}'.format(code))
ax2.axvline(4206, color='black', linestyle='--', lw=1.5, alpha=0.6)
ax2.set_xlabel('순번 (id 앞 4자리)')
ax2.set_ylabel('Shot 번호')
ax2.set_title('순번 x Shot 번호 분포', fontweight='bold')
ax2.legend(markerscale=3, ncol=4, loc='upper left')
ax2.grid(linestyle='--', alpha=0.3)

# -------------------------------------------------------
# 4. 생산코드별 불량률
# -------------------------------------------------------
ax3 = fig.add_subplot(4, 1, 4)
codes = sorted(df_all['생산코드'].unique())
defect_rates = [df_all[df_all['생산코드'] == c]['Defect_Status'].mean() * 100 for c in codes]
bar_colors = [colors.get(c, 'gray') for c in codes]
x = np.arange(len(codes))
bars = ax3.bar(x, defect_rates, color=bar_colors, alpha=0.85, width=0.6)
ax3.set_xticks(x)
ax3.set_xticklabels(['코드 {}'.format(c) for c in codes])
ax3.set_ylabel('불량률 (%)')
ax3.set_title('생산코드별 불량률', fontweight='bold')
ax3.grid(axis='y', linestyle='--', alpha=0.4)
for bar, rate in zip(bars, defect_rates):
    ax3.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
             '{:.1f}%'.format(rate), ha='center', fontsize=9)
ax3.axvline(4.5, color='black', linestyle='--', lw=1.5)
ymax = ax3.get_ylim()[1]
ax3.text(2.0, ymax * 0.92, 'Type1', ha='center', fontsize=10,
         color='#E74C3C', fontweight='bold')
ax3.text(5.5, ymax * 0.92, 'Type2', ha='center', fontsize=10,
         color='#3498DB', fontweight='bold')

plt.tight_layout()
plt.savefig('id_구조_시각화.png', dpi=150, bbox_inches='tight')
print('저장 완료: id_구조_시각화.png')
plt.show()
