import matplotlib.pyplot as plt
import numpy as np

# 绘制速度对比图
fig = plt.figure()
ax = plt.axes()
ax.set(xlim=(0, 40), ylim=(65, 85), xlabel='FPS', ylabel='Vdieo mAP 0.5(%)')
plt.plot(25, 77.2, marker='s', color='g', markersize='10')
plt.plot(20, 80.2, marker='o', color='r', markersize='10')
plt.plot(17, 78.3, marker='^', color='y', markersize='10')
plt.plot(25, 74.7, marker='*', color='c', markersize='10')
plt.plot(25, 73.7, marker='H', color='b', markersize='10')
plt.plot(28, 72.0, marker='p', color='pink', markersize='10')
plt.plot(4, 71.5, marker='D', color='purple', markersize='10')
plt.text(25+1, 77.2, 'MOC', verticalalignment='center', horizontalalignment='center')
plt.text(20+1, 80.2, 'Ours', verticalalignment='center', horizontalalignment='center')
plt.text(17+1, 78.3, 'SAMOC', verticalalignment='center', horizontalalignment='center')
plt.text(25+1, 74.7, '2in1', verticalalignment='center', horizontalalignment='center')
plt.text(25+1, 73.7, 'ACT', verticalalignment='center', horizontalalignment='center')
plt.text(28+1, 72.0, 'Singh et.al', verticalalignment='center', horizontalalignment='center')
plt.text(5, 71.5, 'Saha et.al', verticalalignment='center', horizontalalignment='center')
plt.show()
# plt.savefig('runtime comparison.png')

# 绘制速度分析图
# labels = ['K=3', 'K=5', 'K=7', 'K=9']
# frame_map = [68.81 - 20, 69.74 - 20, 70.78 - 20, 70.87 - 20]
# video_map = [74.27 - 20, 75.57 - 20, 77.10 - 20, 77.74 - 20]
# index = np.arange(len(labels))
# width = 0.3
# fig = plt.figure()
# ax1 = fig.add_subplot()
# rects1 = ax1.bar(index - width / 2, frame_map, width, label='Frame mAP 0.5', color='green', bottom=20)
# rects2 = ax1.bar(index + width / 2, video_map, width, label='Video mAP 0.2', color='lightgreen', bottom=20)
# ax1.set_ylabel('mAP(%)')
# ax1.set_xticks(index)
# ax1.set_xticklabels(labels)
#
# ax2 = ax1.twinx()
# fps = [104, 96, 72, 69]
# rects3, = ax2.plot(index, fps, 'ro-', alpha=0.5, linewidth=2, label='FPS')
# ax2.set_ylim(50, 110)
# ax2.set_ylabel('Frame/second')
# for x, y in zip(index, fps):
#     ax2.text(x, y + 0.5, '%.00f' % y, ha='center', va='bottom')
#
# plt.legend(handles=[rects1, rects2, rects3], bbox_to_anchor=(0.93, 1.12), ncol=3)
# plt.show()
# plt.savefig('runtime analysis.png')
