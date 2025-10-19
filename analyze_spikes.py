import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import argparse
import itertools
from collections import defaultdict
from shapely.geometry import Polygon, Point
from matplotlib.font_manager import FontProperties
# from matplotlib import rc
# #rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
# rc('font',**{'family':'serif','serif':['Times']})

class SpikeInfo:
    rally_dir: Path
    spike_pos3d: list
    adjusted_spike_pos3d: list
    land_pos3d: list
    adjusted_land_pos3d: list
    direction: int
    hit_zone: int
    elapsed_frames: int
    average_speed_ms: float

    def __repr__(self):
        try:
            return f"SpikeInfo({self.rally_dir}, {self.spike_pos3d}, {self.land_pos3d}, {self.direction}, {self.hit_zone}, {self.elapsed_frames})"
        except:
            return "SpikeInfo()"

def draw_volleyball_court(ax):
    # Volleyball court x = 9m, y = 18m
    # Net is along x axis
    # Draw lines
    ax.plot([0, 9], [0, 0], [0, 0], color='black')
    ax.plot([0, 9], [18, 18], [0, 0], color='black')
    ax.plot([0, 0], [0, 18], [0, 0], color='black')
    ax.plot([9, 9], [0, 18], [0, 0], color='black')
    # Draw net
    ax.plot([0, 9], [9, 9], [0, 0], color='black')
    # Draw 3m line
    ax.plot([0, 9], [6, 6], [0, 0], color='black')
    ax.plot([0, 9], [12, 12], [0, 0], color='black')
    # Draw Net
    net_top_height = 2.43
    net_height = 1.0
    ax.plot([0, 9], [9, 9], [net_top_height, net_top_height], color='black')
    ax.plot([0, 9], [9, 9], [net_top_height - net_height, net_top_height - net_height], color='black')
    ax.plot([0, 0], [9, 9], [net_top_height, 0], color='black')
    ax.plot([9, 9], [9, 9], [net_top_height, 0], color='black')
    ax.plot([0, 9], [9, 9], [3.0, 3.0], color='green')

def classify_spike(spike_pos3d, collide_pos3d):
    # Classify the spike
    # 1. Spike direction, 1=lower team, 2=upper team
    spike_direction_vec = np.array(collide_pos3d) - np.array(spike_pos3d)
    spike_direction = spike_direction_vec @ np.array([0, 1, 0])
    if spike_direction > 0:
        spike_direction = 1
    else:
        spike_direction = 2
    # 2. Spike interval
    if spike_direction == 1:
        spike_intervals = [[0, 3], [3, 6], [6, 9]]  # left, middle, right (view from lower team)
    else:
        spike_intervals = [[6, 9], [3, 6], [0, 3]]
    hit_zone = None
    for i, interval in enumerate(spike_intervals):
        if interval[0] <= spike_pos3d[0] < interval[1]:
            hit_zone = i
            break
    return spike_direction, hit_zone

def set_to_half_court(ax):
    ax.set_xlim(0, 9)
    ax.set_ylim(0, 9)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_aspect('equal')

def draw_attack_chart(spike_infos, is_eng=False, font_prop=None):
    # use matplotlib to draw the spike histogram
    team_spikes = defaultdict(list)
    team_spikes[1] = []
    team_spikes[2] = []
    team_spike_arrow_plots = {}
    for spike_info in spike_infos:
        spike_direction = spike_info.direction
        team_spikes[spike_direction].append(spike_info)
    team_hit_location_spikes = {}
    for team_id, spikes in team_spikes.items():
        # hit_location_spikes = defaultdict(list)
        hit_location_spikes = {_:list() for _ in range(3)}

        for spike_info in spikes:
            spike_info: SpikeInfo
            hit_zone = spike_info.hit_zone
            hit_location_spikes[hit_zone].append(spike_info)
        fig, ax = plt.subplots()
        team_spike_arrow_plots[team_id] = (fig, ax)
        # Draw square grid with 3m interval
        for i in range(1, 4):
            ax.plot([0, 9], [i * 3, i * 3], color='black')
            ax.plot([i * 3, i * 3], [0, 9], color='black')
        if is_eng:
            ax.set_title(f"Team {team_id} Attack Chart")
        else:
            # ax.set_title(f"隊伍 {team_id} 攻擊圖表", fontproperties=font_prop)
            ax.set_title(f"攻擊成功圖表", fontproperties=font_prop)
        print(f"Team {team_id} attack chart")
        hit_zone_centers = [1.5, 4.5, 7.5]
        for i, (hit_zone, spikes) in enumerate(hit_location_spikes.items()):
            hit_zone_center = hit_zone_centers[hit_zone]
            hit_zone_pt = [hit_zone_center, 0, 3]
            land_pos3d_list = []
            for spike_info in spikes:
                land_pos3d = spike_info.adjusted_land_pos3d
                if land_pos3d[1] < 0 or land_pos3d[1] > 9:
                    print(f"Outside {land_pos3d}")
                    continue
                if land_pos3d[0] < 0 or land_pos3d[0] > 9:
                    print(f"Outside {land_pos3d}")
                    continue
                land_pos3d_list.append(land_pos3d)
                # Draw arrows
                # Color = hit zone
                color = plt.cm.get_cmap('rainbow')(hit_zone / 5)
                # Scale = larger if faster
                scale = 0.2 + max(1.3 * (spike_info.average_speed_ms - 30) / 110, 0)
                ax.arrow(hit_zone_pt[0], hit_zone_pt[1], land_pos3d[0] - hit_zone_pt[0], land_pos3d[1] - hit_zone_pt[1], \
                         head_width=0.3, head_length=0.3, fc=color, ec=color, width=scale)
                print(f"Hit zone {hit_zone}, spike {land_pos3d}")
        ax.set_xlim(0, 9)
        ax.set_ylim(0, 9)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_aspect('equal')
        team_hit_location_spikes[team_id] = hit_location_spikes

    anchor_pts = np.array([
        [0, 0], [0, 4.5], [0, 9], [4.5, 9], [9, 9], [9, 4.5], [9, 0], [4.5, 0]
    ])
    hit_zone_land_zone_polygon_pt_idxs = [
        [(0, 2, 3), (0, 3, 4, 5), (0, 5, 6)],
        [(7, 0, 2), (7, 2, 4), (7, 4, 6)],
        [(6, 0, 1), (6, 1, 2, 3), (6, 3, 4)]
    ]
    hit_zone_land_zone_polygon_pts = []
    for land_zone_polygon_pt_idxs in hit_zone_land_zone_polygon_pt_idxs:
        pts = []
        for polygon_pts_idxs in land_zone_polygon_pt_idxs:
            polygon_pts = [anchor_pts[i] for i in polygon_pts_idxs]
            pts.append(polygon_pts)
        hit_zone_land_zone_polygon_pts.append(pts)
    
    hit_zone_land_zone_polygon_pts[0] = [
        [(1.5,0), (0,0), (0,9), (1,9)],
        [(1.5,0), (1,9), (4.5,9)],
        [(1.5,0), (4.5,9), (6.75,9)],
        [(1.5,0), (6.75,9), (9,9), (9,6.75)],
        [(1.5,0), (9,6.75), (9,4.5)],
        [(1.5,0), (9,4.5), (9,0)]
    ]
    # flip the x axis
    hit_zone_land_zone_polygon_pts[2] = [[(9 - pt[0], pt[1]) for pt in poly] for poly in hit_zone_land_zone_polygon_pts[0]]
    
    hit_zone_land_zone_polygons = []
    for hit_zone_land_zone_polygon_pts in hit_zone_land_zone_polygon_pts:
        # one polygon for each land zone
        # hit zone have a lot of land zone to attack
        polygons = []
        for polygon_pts in hit_zone_land_zone_polygon_pts:
            polygons.append(Polygon(polygon_pts))
        hit_zone_land_zone_polygons.append(polygons)
    team_spike_angle_plots = {}
    for team_id, spikes in team_spikes.items():
        # Draw histogram for each hit zone
        # Categorize by 3 land zones
        hit_location_spikes = team_hit_location_spikes[team_id]
        if is_eng:
            hit_zone_str = ['Left', 'Middle', 'Right']
        else:
            hit_zone_str = ['左側', '中間', '右側']
        fig = plt.figure()
        team_spike_angle_plots[team_id] = fig
        for hit_zone, spikes in hit_location_spikes.items():
            ax = fig.add_subplot(1, 3, hit_zone + 1)
            set_to_half_court(ax)
            if is_eng:
                ax.set_title(f"Team {team_id} - {hit_zone_str[hit_zone]} Attacks")
            else:
                # ax.set_title(f"隊伍 {team_id} - {hit_zone_str[hit_zone]}攻擊", fontproperties=font_prop)
                ax.set_title(f"{hit_zone_str[hit_zone]}攻擊", fontproperties=font_prop)
            land_zone_counts = [0] * len(hit_zone_land_zone_polygons[hit_zone])
            land_zone_polygons = hit_zone_land_zone_polygons[hit_zone]
            for spike_info in spikes:
                land_pos3d = spike_info.adjusted_land_pos3d
                for i, hit_zone_polygon in enumerate(land_zone_polygons):
                    if hit_zone_polygon.contains(Point(land_pos3d[0], land_pos3d[1])):
                        land_zone_counts[i] += 1
                        break
            # draw the polygon and fill blue color which the count number larger the deeper, and put count number on middle
            # color_map = plt.cm.get_cmap('Blues', 10)
            num_colors = 10
            color_map = matplotlib.colormaps.get_cmap('Blues')
            for i, land_zone_polygon in enumerate(land_zone_polygons):
                color = color_map(land_zone_counts[i] / max(1, sum(land_zone_counts)) * 0.8) # * 0.8 to make it lighter, prevent text not visible
                ax.fill(*land_zone_polygon.exterior.xy, color=color, alpha=1.0)
                # Draw the shape line of polygon
                ax.plot(*land_zone_polygon.exterior.xy, color=[c/255 for c in (150, 150, 150)], linewidth=0.5)
                # This text is not good, it will cross the polygon
                # ax.text(land_zone_polygon.centroid.x, land_zone_polygon.centroid.y, str(land_zone_counts[i]), fontsize=12)
                ax.text(land_zone_polygon.centroid.x, land_zone_polygon.centroid.y, str(land_zone_counts[i]), fontsize=12, ha='center', va='center', color='black')
    return team_spike_arrow_plots, team_spike_angle_plots

def main():
    parser = argparse.ArgumentParser(description='Analyze spikes from rallies')
    parser.add_argument('--gamedir', type=str, default="./stream_results/stream_result_32min", help='Game dir path')
    parser.add_argument('--outdir', default=None, help='Output dir')
    parser.add_argument('--fps', type=int, default=60, help='FPS')
    parser.add_argument('--show', action='store_true', help='Show the plot')
    parser.add_argument('--eng', action='store_true', help='English mode')
    args = parser.parse_args()
    is_eng = args.eng
    fps = args.fps
    game_dir = Path(args.gamedir)
    if not game_dir.exists():
        print(f"Rally dir {game_dir} does not exist")
        return
    MAX_BALL_MS = 140 / 3.6  # 120 km/h
    MIN_BALL_MS = 30 / 3.6  # 50 km/h
    # MAX_HALF_COURT_LENGTH = (9**2 + 9**2) ** 0.5
    MAX_HALF_COURT_LENGTH = 18
    rally_dirs = [d for d in game_dir.iterdir() if d.is_dir()]
    rally_dirs.sort()
    spike_infos = []
    total_spikes = 0
    total_attacks = 0
    total_collides = 0
    total_jsons = 0
    total_filter_ball_not_on_ground = 0

    font_prop = FontProperties()
    font_prop.set_file('NotoSansTC-Medium.ttf')
    font_prop.set_size(16)

    for rally_dir in rally_dirs:
        print(f"=== Analyzing rally {rally_dir} ===")
        rally_name = rally_dir.name
        json_path = rally_dir / 'ball_data.json'
        if not json_path.exists():
            print(f"Cannot find ball data json file in {rally_dir}")
            continue
        total_jsons += 1
        with open(json_path, 'r') as f:
            rally_data = json.load(f)
        ball_data = {}
        for frame_data in rally_data['ball_data']:
            frame_id = frame_data['frame_id']
            ball_data[frame_id] = frame_data
        
        spikes = [data for data in rally_data['event'] if data['event'] == 'spike']
        attacks = [data for data in rally_data['event'] if data['event'] == 'attack']
        collides = [data for data in rally_data['event'] if data['event'] == 'collide']
        print(f"Spike count: {len(spikes)}, Collide count: {len(collides)}")
        total_spikes += len(spikes)
        total_attacks += len(attacks)
        for spike in spikes + attacks:
            spike_frame_id = spike['frame_id']
            spike_ball_data = ball_data[spike_frame_id]
            spike_pos3d = spike_ball_data['pos3d']
            # find the collide after the spike
            spike_land_collide = None
            for collide in collides:
                if collide['frame_id'] > spike_frame_id:
                    print("Frame diff", collide['frame_id'] - spike_frame_id)
                if (2 < (collide['frame_id'] - spike_frame_id) < (MAX_HALF_COURT_LENGTH / MIN_BALL_MS * fps + 3)):
                    spike_land_collide = collide
                    break
            print("max frame diff", MAX_HALF_COURT_LENGTH / MIN_BALL_MS * fps)
            if spike_land_collide is None:
                print(f">> Fail to find collide after spike at frame {spike_frame_id}")
                end_search_frame_id = spike_frame_id + 30
            else:
                end_search_frame_id = spike_land_collide['frame_id']

            interval_pos3ds = [ball_data[frame_id]['pos3d'] for frame_id in range(spike_frame_id, end_search_frame_id + 1) \
                               if frame_id in ball_data]
            interval_frame_ids = [frame_id for frame_id in range(spike_frame_id, end_search_frame_id + 1) \
                                  if frame_id in ball_data]
            interval_zs = [pos3d[2] for pos3d in interval_pos3ds]
            min_z = min(interval_zs)
            min_z_idx = interval_zs.index(min_z)
            if min_z > 1.5: # 50 cm
            # if min_z > 2.0: # 50 cm
                print(f">> Ball is not on the ground, min_z = {min_z:.2f}")
                total_filter_ball_not_on_ground += 1
                continue
            elapsed_frames = (interval_frame_ids[min_z_idx] - spike_frame_id)
            land_pos3d = interval_pos3ds[min_z_idx]
            si = SpikeInfo()
            si.rally_dir = rally_dir
            si.spike_pos3d = spike_pos3d
            si.land_pos3d = land_pos3d
            si.elapsed_frames = elapsed_frames
            si.average_speed_ms = (np.linalg.norm(np.array(spike_pos3d) - np.array(land_pos3d)) / elapsed_frames) * fps
            spike_infos.append(si)
    print(f"Total spikes: {total_spikes}/{total_jsons}")
    print(f"Total attacks: {total_attacks}/{total_jsons}")
    print(f"Total filter ball not on ground: {total_filter_ball_not_on_ground}")
    print(f"Total spike paths: {len(spike_infos)}")
    # print('\n'.join([str(spike_path) for spike_path in spike_infos]))
    # Plot the 3D spikes
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    draw_volleyball_court(ax)
    spike_intervals = [[0, 3], [3, 6], [6, 9]]
    spike_directions = [1, 2]
    # colormap = plt.cm.get_cmap('rainbow', len(spike_intervals) * len(spike_directions))
    colormap = matplotlib.colormaps.get_cmap('rainbow')
    num_colors = len(spike_intervals) * len(spike_directions)
    valid_spike_infos = []
    for spike_info in spike_infos:
        spike_info: SpikeInfo
        # rally_dir, spike_pos3d, collide_pos3d = spike_path
        rally_dir = spike_info.rally_dir
        spike_pos3d = spike_info.spike_pos3d
        land_pos3d = spike_info.land_pos3d

        spike_direction, hit_zone = classify_spike(spike_pos3d, land_pos3d)
        if hit_zone is None:
            continue
        valid_spike_infos.append(spike_info)

        spike_info.direction = spike_direction
        spike_info.hit_zone = hit_zone
        # adjusted position = use net as bottom (y=0) and toward positive y
        if spike_direction == 1:
            spike_info.adjusted_spike_pos3d = [spike_pos3d[0], spike_pos3d[1] - 9, spike_pos3d[2]]
            spike_info.adjusted_land_pos3d  = [land_pos3d[0],  land_pos3d[1] - 9,  land_pos3d[2]]
        else:
            spike_info.adjusted_spike_pos3d = [9 - spike_pos3d[0], 9 - spike_pos3d[1], spike_pos3d[2]]
            spike_info.adjusted_land_pos3d  = [9 - land_pos3d[0],  9 - land_pos3d[1],  land_pos3d[2]]
        color = colormap((hit_zone * 2 + spike_direction - 1) / num_colors)
        ax.plot([spike_pos3d[0], land_pos3d[0]], [spike_pos3d[1], land_pos3d[1]], [spike_pos3d[2], land_pos3d[2]], color=color)
    spike_infos = valid_spike_infos
    # for spike_path in spike_paths:
    #     rally_dir, spike_pos3d, collide_pos3d = spike_path
    #     ax.plot([spike_pos3d[0], collide_pos3d[0]], [spike_pos3d[1], collide_pos3d[1]], [spike_pos3d[2], collide_pos3d[2]])
    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')
    # set the xyz the same scale
    ax.set_xlim(4.5-9, 4.5+9)
    ax.set_ylim(0, 18)
    ax.set_zlim(-9, 9)
    team_spike_arrow_plots, team_spike_angle_plots = draw_attack_chart(spike_infos, is_eng=is_eng, font_prop=font_prop)
    if args.show:
        plt.show()
    else:
        if args.outdir is not None:
            outdir = Path(args.outdir)
            outdir.mkdir(exist_ok=True)
        else:
            outdir = game_dir
        fig.savefig(outdir / "spike_paths.png")
        for team_id, (fig, ax) in team_spike_arrow_plots.items():
            fig.tight_layout()
            fig.savefig(outdir / f"team_{team_id}_attack_arrow.png")

        # all_team_fig = plt.subplot()
        num_teams = len(team_spike_angle_plots)
        for team_id, fig in team_spike_angle_plots.items():
            fig.tight_layout()
            fig.savefig(outdir / f"team_{team_id}_attack_angle.png")
            # all_team_fig.add_subplot(num_teams, 1, team_id)
            # all_team_fig.imshow(fig)
        

if __name__ == '__main__':
    main()