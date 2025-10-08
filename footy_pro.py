# footy_pro_v6.py
# Pygame 11v11 football sim:
# - true holder possession (no teleport)
# - tackles/duels -> fouls (immediate whistle; no advantage)
# - direct/indirect FKs w/ wall; corners/throws patterns
# - proper offside (second phase; deliberate defender play resets, GK save does NOT)
# - GK: 6-second + back-pass, sweeping & distribution
# - interceptions, injuries + forced subs
# - VAR/GLT overlay for tight calls
# - multi-formations (4-3-3, 4-2-3-1, 3-5-2), roles, chips/crosses (ball z), collisions
# - time-wasting & gamble heuristics; HUD with speed controls

import math, random, time, sys, importlib.util
from dataclasses import dataclass, field
from typing import List, Tuple, Optional

if importlib.util.find_spec("pygame") is None:
    sys.stderr.write(
        "football simulator requires the optional dependency 'pygame'.\n"
        "Install it with `pip install pygame` to launch the full experience.\n"
    )
    raise SystemExit(1)

import pygame

# =================== Pitch & physics ===================
PITCH_LEN_M, PITCH_WID_M = 105.0, 68.0
BOX_18_M, BOX_6_M, SPOT_M, CIRCLE_R = 16.5, 5.5, 11.0, 9.15

# Carry/possession & ball
CARRY_OFFSET = 16.0      # px ahead of holder
DRIBBLE_WOBBLE = 4.0     # px variation while carrying
CAPTURE_RADIUS = 14.0    # px to pick up a loose slow ball
CAPTURE_SPEED  = 14.0    # px/s under this is capturable
BALL_FRICTION_GROUND = 0.985
GRAVITY = 980.0          # px/s^2 (scaled; only for airborne)
BOUNCE_DAMP  = 0.55

# Duels / fouls (immediate whistle; no advantage)
TACKLE_RADIUS_PX    = 28.0
FOUL_BASE_RATE      = 0.08
SECOND_YELLOW_CHANCE = 0.28
PERSISTENT_THRESHOLD = 3

# Keeper
GK_SWEEP_TRIGGER_XM = 24.0
GK_MAX_SPEED_BONUS  = 1.15
GK_HOLD_MAX_SECONDS = 6.0

# Restarts & set pieces
WALL_DISTANCE_M     = 9.15
CORNER_CROSS_POWER  = 280
THROW_FLICK_POWER   = 200

# Collisions
COLLISION_RADIUS_PX = 20.0

# UI / sim speed (seconds per in-game minute)
SIM_SPEEDS = [0.6, 1.05, 1.8]
STOPPAGE_PAUSE = 1.6

# =================== Models ===================
@dataclass(eq=False)
class Player:
    name: str; pos: str; number: int
    atk: int; dfn: int; pas: int; pac: int; sta: int; dis: int; gk: int=1
    fatigue: float=0.0; yellow: bool=False; red: bool=False; injured: bool=False; on_pitch: bool=True
    x: float=0.0; y: float=0.0; vx: float=0.0; vy: float=0.0
    role: str=""      # archetype hint
    foot: str="both"  # "left","right","both"
    fouls_committed: int=0
    injury_timer: float=0.0
    used_sub: bool=False

    def o_atk(self):
        f = max(0.5, 1.0 - self.fatigue)
        return (0.5*self.atk + 0.3*self.pas + 0.2*self.pac) * f

    def o_dfn(self):
        f = max(0.5, 1.0 - self.fatigue)
        return (0.55*self.dfn + 0.25*self.pas + 0.20*self.pac) * f

    def tick_fatigue(self, intensity: float):
        drain = 0.006 * intensity * (100 - (0.6*self.sta + 0.4*self.pac)) / 100
        self.fatigue = min(0.7, self.fatigue + max(0.0, drain))

@dataclass(eq=False)
class Team:
    name: str; color: Tuple[int,int,int]; alt: Tuple[int,int,int]
    players: List[Player]; subs: List[Player]
    tactic: str="balanced"; formation: str="4-3-3"; manager: str="Manager"
    goals:int=0; shots:int=0; shots_on:int=0; xg:float=0.0; yellows:int=0; reds:int=0
    benched: List[Player] = field(default_factory=list)


    def gk(self):
        for p in self.players:
            if p.pos=="GK" and p.on_pitch and not p.red and not p.injured: return p
        return next(p for p in self.players if p.on_pitch)

    def outfield(self):  return [p for p in self.players if p.pos!="GK" and p.on_pitch and not p.red and not p.injured]

    def atk_rating(self):
        of=self.outfield()
        base=(sum(p.o_atk() for p in of)/len(of)) if of else 40
        mult={"defensive":0.94,"balanced":1.0,"attacking":1.12,"press":1.07}[self.tactic]
        return base*mult

    def dfn_rating(self):
        of=self.outfield()
        base=(sum(p.o_dfn() for p in of)/len(of)) if of else 40
        mult={"defensive":1.10,"balanced":1.0,"attacking":0.95,"press":1.02}[self.tactic]
        return base*mult

    def pass_skill(self):
        of=self.outfield()
        return (sum(p.pas for p in of)/len(of)) if of else 55

    def discipline(self):
        of=self.outfield()
        return (sum(p.dis for p in of)/len(of)) if of else 60

    def fatigue_tick(self):
        intensity={"defensive":0.9,"balanced":1.0,"attacking":1.06,"press":1.18}[self.tactic]
        for p in self.outfield():
            p.tick_fatigue(intensity)

    def make_substitution(self, off: Player, sub: Player) -> bool:
        """Swap an on-pitch player for a bench option, keeping player list order."""
        if off not in self.players:
            return False
        if sub not in self.subs or sub.on_pitch:
            return False

        try:
            idx = self.players.index(off)
        except ValueError:
            return False

        # remove the incoming player from any bench tracking collections
        if sub in self.subs:
            self.subs.remove(sub)
        if sub in self.benched:
            self.benched.remove(sub)

        # activate the substitute
        sub.on_pitch = True
        sub.injured = False
        sub.used_sub = True
        sub.x, sub.y = off.x, off.y
        sub.vx = sub.vy = 0.0

        # place the substitute into the players list at the original index
        self.players[idx] = sub

        # move the replaced player to the bench
        off.on_pitch = False
        off.vx = off.vy = 0.0
        off.used_sub = True
        if off not in self.subs:
            self.subs.append(off)
        if off not in self.benched:
            self.benched.append(off)
        return True

# =================== Formations & anchors ===================
def anchors_433(left=True):
    return ([(5,34),(18,15),(22,27),(22,41),(18,53),(35,34),(45,22),(45,46),(70,16),(73,34),(70,52)]
            if left else [(100,34),(87,53),(83,41),(83,27),(87,15),(70,34),(60,46),(60,22),(35,52),(32,34),(35,16)])

def anchors_4231(left=True):
    return ([(5,34),(18,16),(22,26),(22,42),(18,52),(34,40),(34,28),(50,28),(50,40),(50,52),(73,34)]
            if left else [(100,34),(87,52),(83,42),(83,26),(87,16),(71,40),(71,28),(55,40),(55,28),(55,16),(32,34)])

def anchors_352(left=True):
    return ([(5,34),(22,20),(22,34),(22,48),(36,24),(36,44),(52,34),(68,22),(68,46),(76,30),(76,38)]
            if left else [(100,34),(83,48),(83,34),(83,20),(69,44),(69,24),(53,34),(37,46),(37,22),(29,38),(29,30)])

def anchors_for(form, left):     
    return {"4-3-3":anchors_433, "4-2-3-1":anchors_4231, "3-5-2":anchors_352}.get(form, anchors_433)(left)

# =================== Team builder ===================
def gen_player(name,pos,num, base=65, spread=12, gk=False, role="", foot="both"):
    r=lambda: max(35, min(95, int(random.gauss(base, spread))))
    if gk:   return Player(name,pos,num,40,50,r(),r(),r(),r(),gk=r(), role="sweeper", foot=foot)
    if pos=="DF": return Player(name,pos,num,r()-10,r()+10,r(),r(),r(),r(), role=("bpd" if random.random()<0.35 else "cb"), foot=foot)
    if pos=="MF": return Player(name,pos,num,r(),r(),r()+8,r(),r(),r(), role=("anchor" if random.random()<0.3 else "cm"), foot=foot)
    if pos=="FW": return Player(name,pos,num,r()+10,r()-10,r(),r()+5,r(),r(), role=("target" if random.random()<0.35 else "winger"), foot=foot)
    return Player(name,pos,num,r(),r(),r(),r(),r(),r(), role=role, foot=foot)

def make_team(name, seed, color, alt):
    random.seed(seed); nums=iter(range(1,12))
    starters=[
        gen_player(f"{name} GK","GK",next(nums),base=69,gk=True,foot="right"),
        gen_player(f"{name} DF1","DF",next(nums),base=66,foot="left"),
        gen_player(f"{name} DF2","DF",next(nums),base=66),
        gen_player(f"{name} DF3","DF",next(nums),base=66),
        gen_player(f"{name} DF4","DF",next(nums),base=66,foot="right"),
        gen_player(f"{name} MF1","MF",next(nums),base=70,role="anchor"),
        gen_player(f"{name} MF2","MF",next(nums),base=70),
        gen_player(f"{name} MF3","MF",next(nums),base=70,role="am"),
        gen_player(f"{name} FW1","FW",next(nums),base=72,role="winger",foot="left"),
        gen_player(f"{name} FW2","FW",next(nums),base=72,role="target"),
        gen_player(f"{name} FW3","FW",next(nums),base=72,role="winger",foot="right"),
    ]
    for p in starters: p.on_pitch=True
    bench=[
        gen_player(f"{name} SUB1","DF",12,base=65),
        gen_player(f"{name} SUB2","MF",13,base=67),
        gen_player(f"{name} SUB3","MF",14,base=67),
        gen_player(f"{name} SUB4","FW",15,base=69),
        gen_player(f"{name} SUB5","DF",16,base=65),
    ]
    for s in bench: s.on_pitch=False
    return Team(name,color,alt,starters,bench,"balanced","4-3-3",f"{name} Manager")

# =================== Manager AI ===================
class ManagerAI:
    def __init__(self, team: Team, home: bool):
        self.t=team; self.home=home; self.subs_used=0; self.max_subs=5; self.last_min=-999

    def _bench_candidates(self):
        pool=[]; seen=set()
        for group in (self.t.subs, self.t.benched):
            for s in group:
                if s.on_pitch or s.used_sub:
                    continue
                key=id(s)
                if key in seen:
                    continue
                seen.add(key)
                pool.append(s)
        return pool

    @staticmethod
    def _best_option(pool):
        return max(pool, key=lambda s: (s.sta + s.pac + s.pas), default=None)

    def decide(self, minute, diff, my_xg, opp_xg, injuries: List[Player]):
        # formation/tactic switches
        if minute-self.last_min>=6:
            chasing=diff<0; leading=diff>0; late=minute>=75; mid=35<=minute<75
            if chasing and late: self.t.tactic="attacking"; self.t.formation=random.choice(["4-2-3-1","3-5-2"])
            elif chasing and mid: self.t.tactic="press"; self.t.formation="4-3-3"
            elif leading and late: self.t.tactic="defensive"; self.t.formation=random.choice(["4-2-3-1","4-3-3"])
            else:
                tilt=my_xg-opp_xg
                if tilt<-0.5 and minute>=50: self.t.tactic="press"
                elif tilt>0.7 and minute>=60: self.t.tactic="balanced"
            self.last_min=minute

        # forced subs for injuries
        for p in injuries:
            if p.on_pitch and p.injured and self.subs_used < self.max_subs:
                bench = self._bench_candidates()
                # try to match position first
                positional = [s for s in bench if s.pos == p.pos]
                pool = positional or bench
                if not pool:
                    continue
                sub = self._best_option(pool)
                if sub and self.t.make_substitution(p, sub):
                    self.subs_used += 1

        # planned subs
        if self.subs_used < self.max_subs and minute >= 58:
            tired = sorted(
                [p for p in self.t.players if p.on_pitch and p.pos != "GK"],
                key=lambda q: q.fatigue + (0.15 if q.yellow else 0),
                reverse=True,
            )
            bench = self._bench_candidates()
            if tired and bench:
                off = tired[0]
                need_att = diff < 0 and minute >= 65
                need_def = diff > 0 and minute >= 70

                def line(p):
                    return p.pos if p.pos in ("DF", "MF", "FW") else "MF"

                pool = [s for s in bench if line(s) == line(off)]
                if need_att:
                    att_pool = [s for s in bench if s.pos == "FW"]
                    if att_pool:
                        pool = att_pool
                if need_def:
                    def_pool = [s for s in bench if s.pos == "DF"]
                    if def_pool:
                        pool = def_pool
                if not pool:
                    pool = bench
                if pool:
                    sub = self._best_option(pool)
                    if sub and self.t.make_substitution(off, sub):
                        self.subs_used += 1
                        self.last_min = minute
def clamp(v,a,b): return max(a, min(b, v))

class Match:
    def __init__(self, H: Team, A: Team, seed=1234):
        random.seed(seed)
        self.H, self.A = H, A
        self.aiH, self.aiA = ManagerAI(H,True), ManagerAI(A,False)
        self.minute=0; self.half=1; self.stoppage=0
        self.poss = self.H if random.random()<0.55 else self.A
        # ball (x,y,z) & velocity (vx,vy,vz)
        self.ball=[0.0,0.0,0.0]; self.bvx=0.0; self.bvy=0.0; self.bvz=0.0
        self.pr: Optional[pygame.Rect]=None
        self.holder: Optional[Player]=None
        self.last_touch_team=None; self.last_touch_player=None
        self.restart=None; self.deadball_timer=0.0
        # offside phase
        self.phase_reset = True
        # GK hands state
        self.gk_hands=False; self.gk_hold_timer=0.0
        # VAR overlay
        self.var_timer=0.0; self.var_text=""
        # logs
        self.fouls_log = {}
        self.last_pass = {"team":None, "passer":None, "receiver":None, "type":"ground"}
        self.injuries_to_process: List[Player]=[]
        self.pass_target: Optional[Player] = None
        self.commentary: List[str] = []
        self.commentary_limit = 6
        self.referee = {"x": 0.0, "y": 0.0, "vx": 0.0, "vy": 0.0}
        self.manager_positions = {}
        self.bench_spots = {}
        self.dribble_phase = 0.0
        self.release_player: Optional[Player] = None
        self.release_timer = 0.0
        self.goal_flash_timer = 0.0
        self.goal_flash_duration = 0.0
        self.goal_flash_text = ""
        self.goal_flash_color = (255, 255, 255)
        self.in_stoppage = False

    # ---------- helpers ----------
    def add_commentary(self, text: str):
        stamp = f"{self.minute:02d}'"
        entry = f"{stamp} {text}"
        self.commentary.append(entry)
        if len(self.commentary) > self.commentary_limit:
            self.commentary = self.commentary[-self.commentary_limit:]

    def m2p(self, xm, ym):
        x = self.pr.x + (xm/PITCH_LEN_M)*self.pr.w
        y = self.pr.y + (ym/PITCH_WID_M)*self.pr.h
        return x,y

    def reset_for_kickoff(self):
        if not self.pr:
            return
        ah=[self.m2p(*t) for t in anchors_for(self.H.formation, True)]
        aa=[self.m2p(*t) for t in anchors_for(self.A.formation, False)]
        for i,p in enumerate(self.H.players):
            if i < len(ah) and p.on_pitch:
                p.x,p.y=ah[i]
            p.vx=p.vy=0.0
        for i,p in enumerate(self.A.players):
            if i < len(aa) and p.on_pitch:
                p.x,p.y=aa[i]
            p.vx=p.vy=0.0

    def kickoff(self, pr: pygame.Rect):
        self.pr=pr
        self.reset_for_kickoff()
        self.place_ball_center()
        self.holder = min((self.H.players if self.poss is self.H else self.A.players),
                          key=lambda p: (p.x-self.ball[0])**2+(p.y-self.ball[1])**2)
        self.gk_hands=False; self.gk_hold_timer=0.0
        # sideline setup
        outer_offset = max(24, min(max(self.pr.x - 12, 0), 48))
        screen_width = self.pr.right + self.pr.x
        left_margin = self.pr.x
        right_margin = max(screen_width - self.pr.right, 0)
        if left_margin >= 36:
            home_lane = self.pr.x - min(outer_offset, left_margin - 12)
        else:
            home_lane = self.pr.x + 24
        if right_margin >= 36:
            away_lane = self.pr.right + min(outer_offset, right_margin - 12)
        else:
            away_lane = self.pr.right - 24
        self.manager_positions = {
            id(self.H): (home_lane, self.pr.y + self.pr.h * 0.35),
            id(self.A): (away_lane, self.pr.y + self.pr.h * 0.65),
        }

        def bench_slots(team: Team, left: bool):
            lane_x = home_lane if left else away_lane
            start_y = self.pr.y + 90
            step = 28
            return [(lane_x, start_y + i * step) for i in range(12)]

        self.bench_spots = {
            id(self.H): bench_slots(self.H, True),
            id(self.A): bench_slots(self.A, False)
        }

        self.update_bench_positions()
        self.referee["x"], self.referee["y"] = self.ball[0], self.ball[1]
        self.referee["vx"] = self.referee["vy"] = 0.0
        self.add_commentary(f"Kick-off: {self.poss.name} get us underway")

    def place_ball_center(self):
        self.ball[0]=self.pr.x+self.pr.w/2; self.ball[1]=self.pr.y+self.pr.h/2; self.ball[2]=0.0
        self.bvx=self.bvy=self.bvz=0.0

    def nearest(self, team:Team):
        cand=[p for p in team.players if p.on_pitch]
        return None if not cand else min(cand, key=lambda p:(p.x-self.ball[0])**2 + (p.y-self.ball[1])**2)

    def keeper(self, team:Team):  return team.gk()

    def update_bench_positions(self):
        if not self.pr:
            return
        for team in (self.H, self.A):
            spots = self.bench_spots.get(id(team), [])
            benchers = [p for p in team.subs if not p.on_pitch]
            benchers += [p for p in team.benched if not p.on_pitch]
            for idx, player in enumerate(benchers):
                if idx < len(spots):
                    player.x, player.y = spots[idx]
                    player.vx = player.vy = 0.0

    # ---------- possession changes ----------
    def on_kick(self, team:Team, passer:Player, ptype="ground"):
        self.last_touch_team=team; self.last_touch_player=passer
        self.last_pass={"team":team, "passer":passer, "receiver":None, "type":ptype}
        self.holder=None
        # passing begins new offside phase
        self.phase_reset=False
        self.pass_target=None

    def capture_free_ball(self):
        if self.holder is not None or self.restart: return
        speed=math.hypot(self.bvx,self.bvy)
        all_players=[p for T in (self.H,self.A) for p in T.players if p.on_pitch and not p.injured and not p.red]
        contenders = sorted(all_players, key=lambda p:(p.x-self.ball[0])**2+(p.y-self.ball[1])**2)
        for taker in contenders:
            d=math.hypot(taker.x-self.ball[0], taker.y-self.ball[1])
            if (self.release_player is taker and self.release_timer > 0.0 and
                    speed > CAPTURE_SPEED * 1.2 and d < CAPTURE_RADIUS * 1.5):
                continue
            if d <= CAPTURE_RADIUS or speed <= CAPTURE_SPEED:
                self.holder=taker
                self.poss = self.H if taker in self.H.players else self.A
                self.bvx=self.bvy=self.bvz=0.0
                self.release_player=None
                self.release_timer=0.0
                # GK hands if catch in air
                if taker is self.keeper(self.poss) and self.ball[2] > 2.0:
                    self.gk_hands=True; self.gk_hold_timer=0.0
                else:
                    self.gk_hands=False
                if self.pass_target is taker:
                    self.add_commentary(f"{taker.name} brings the pass under control")
                elif self.last_touch_team and taker not in self.last_touch_team.players:
                    self.add_commentary(f"{taker.name} steals the loose ball")
                self.pass_target=None
                # back-pass violation (only if hands used on deliberate ground pass)
                if taker is self.keeper(self.poss) and self.last_pass["team"] is self.poss and self.last_pass["type"]=="ground":
                    if self.gk_hands:
                        self.indirect_free_kick(self.A if self.poss is self.H else self.H, taker.x, taker.y, reason="Back-pass")
                        if not taker.yellow and random.random()<0.5: taker.yellow=True; self.poss.yellows+=1
                return

    # ---------- offsides ----------
    def is_offside(self, atk: Team, receiver: Player):
        if self.phase_reset: return False
        defending = self.A if atk is self.H else self.H
        xs = sorted([p.x for p in defending.players if p.on_pitch])
        if len(xs)<2: return False
        line = xs[1]
        ball_x = self.ball[0]
        if atk is self.H:
            return receiver.x > max(line, ball_x) + 1
        else:
            return receiver.x < min(line, ball_x) - 1

    # ---------- fouls & cards ----------
    def whistle_foul(self, atk:Team, dfn:Team, spotx, spoty, offender:Player):
        offender.fouls_committed += 1
        give_yellow=False; give_red=False
        if offender.yellow and random.random()<SECOND_YELLOW_CHANCE:
            give_red=True
        elif offender.fouls_committed>=PERSISTENT_THRESHOLD or random.random()<0.22:
            give_yellow=True
        if give_red:
            offender.red=True; offender.on_pitch=False; dfn.reds+=1
            self.add_commentary(f"Red card! {offender.name} is sent off")
        elif give_yellow and not offender.yellow:
            offender.yellow=True; dfn.yellows+=1
            self.add_commentary(f"Yellow card for {offender.name}")
        self.direct_free_kick(atk, spotx, spoty)

    def direct_free_kick(self, team:Team, x, y):
        self.bvx=self.bvy=self.bvz=0.0; self.holder=None
        self.restart=('dfk', x, y, team); self.deadball_timer=STOPPAGE_PAUSE
        self.gk_hands=False
        self.pass_target=None
        self.release_player=None
        self.release_timer=0.0
        self.add_commentary(f"Direct free-kick for {team.name}")

    def indirect_free_kick(self, team:Team, x, y, reason="IFK"):
        self.bvx=self.bvy=self.bvz=0.0; self.holder=None
        self.restart=('ifk', x, y, team, reason); self.deadball_timer=STOPPAGE_PAUSE
        self.gk_hands=False
        self.pass_target=None
        self.release_player=None
        self.release_timer=0.0
        self.add_commentary(f"Indirect free-kick for {team.name} ({reason})")

    # ---------- keeper behaviours ----------
    def keeper_positioning(self, team:Team, left_to_right:bool, dt):
        gk=self.keeper(team)
        if not (gk and gk.on_pitch): return
        bx, by = self.ball[0], self.ball[1]
        depth_m = 4.0 if team.tactic=="defensive" else (6.5 if team.tactic=="balanced" else 9.0)
        xm = depth_m if left_to_right else (PITCH_LEN_M - depth_m)
        tx, ty = self.m2p(xm, PITCH_WID_M/2)
        ty = 0.75*ty + 0.25*by
        ms = (58 + 46*(gk.pac/100.0)) * 0.65
        dx, dy = tx-gk.x, ty-gk.y
        d = max(1e-6, math.hypot(dx,dy))
        gk.vx, gk.vy = (dx/d)*ms*dt, (dy/d)*ms*dt
        gk.x += gk.vx; gk.y += gk.vy

    def keeper_sweep_try(self, defending:Team, left_to_right_defending:bool, dt):
        gk=self.keeper(defending)
        if not (gk and gk.on_pitch) or self.holder is not None: return
        bx = self.ball[0]
        trig_xm = GK_SWEEP_TRIGGER_XM if left_to_right_defending else (PITCH_LEN_M - GK_SWEEP_TRIGGER_XM)
        trig_x,_ = self.m2p(trig_xm, PITCH_WID_M/2)
        fast = (abs(self.bvx)+abs(self.bvy)) > 80
        towards_goal = (bx < trig_x) if left_to_right_defending else (bx > trig_x)
        if towards_goal and fast:
            ms = (58 + 46*(gk.pac/100.0)) * GK_MAX_SPEED_BONUS
            dx, dy = self.ball[0]-gk.x, self.ball[1]-gk.y
            d=max(1e-6, math.hypot(dx,dy))
            gk.vx, gk.vy = (dx/d)*ms*dt, (dy/d)*ms*dt
            gk.x += gk.vx; gk.y += gk.vy
            if d < 26:
                self.poss = defending; self.holder=gk
                self.bvx=self.bvy=self.bvz=0.0
                self.gk_hands=True; self.gk_hold_timer=0.0
                self.phase_reset=True  # keeper control is a new phase
                self.pass_target=None
                self.release_player=None
                self.release_timer=0.0
                self.add_commentary(f"{gk.name} sweeps up the danger")

    def keeper_distribute(self, team:Team):
        gk=self.keeper(team)
        if not (gk and gk.on_pitch): return
        mates=[m for m in team.players if m.on_pitch and m is not gk]
        if not mates: return
        losing = (team is self.H and self.H.goals<self.A.goals) or (team is self.A and self.A.goals<self.H.goals)
        long_ball = team.tactic in ('press','attacking') or losing
        if long_ball:
            target=max(mates, key=lambda m:(m.pac+m.sta + (m.atk if m.pos=="FW" else 0)))
            self.pass_to(team, gk, target, power=360, ptype="chip")
        else:
            backs=[m for m in mates if m.pos=='DF']
            target=min(backs or mates, key=lambda m:(m.x-gk.x)**2+(m.y-gk.y)**2)
            self.pass_to(team, gk, target, power=220, ptype="ground")
        self.gk_hands=False; self.gk_hold_timer=0.0
        self.add_commentary(f"{gk.name} distributes for {team.name}")

    # ---------- passing/shooting ----------
    def pass_to(self, team:Team, passer:Player, target:Player, power=320, ptype="ground"):
        if self.is_offside(team, target):
            self.indirect_free_kick(self.A if team is self.H else self.H, self.ball[0], self.ball[1], reason="Offside")
            return
        self.on_kick(team, passer, ptype)
        self.release_player = passer
        self.release_timer = 0.35
        if isinstance(target, Player) and target in team.players and target.on_pitch:
            self.pass_target = target
        ang=math.atan2(target.y-self.ball[1], target.x-self.ball[0])
        if ptype in ("chip","cross"):
            self.bvx, self.bvy = math.cos(ang)*power, math.sin(ang)*power
            self.bvz = 240.0
            self.ball[2] = max(self.ball[2], 1.0)
        else:
            self.bvx, self.bvy = math.cos(ang)*power, math.sin(ang)*power
            self.bvz = 0.0
        self.last_pass["receiver"]=target
        if isinstance(target, Player) and target in team.players:
            desc = "chip" if ptype == "chip" else ("cross" if ptype == "cross" else "pass")
            self.add_commentary(f"{passer.name} plays a {desc} to {target.name}")

    def try_shot(self, atk:Team, dfn:Team, shooter:Player):
        gx = self.pr.right-18 if atk is self.H else self.pr.x+18
        gy = self.pr.y + self.pr.h/2
        dist = max(6.0, math.hypot(shooter.x-gx, shooter.y-gy))
        angle = abs(math.atan2(gy-shooter.y, gx-shooter.x))
        base_xg = max(0.03, min(0.55, 0.32*(1.0/(dist/110.0)) * (1.2 - angle/1.6)))
        delta = (atk.atk_rating()-dfn.dfn_rating())/140.0
        xg = base_xg*(1.0+delta*0.6)*random.uniform(0.85,1.15)
        finesse = random.random()<0.35
        power = 360 if not finesse else 280
        self.on_kick(atk, shooter, ptype="ground")
        self.release_player = shooter
        self.release_timer = 0.35
        ang=math.atan2(gy-self.ball[1], gx-self.ball[0])
        self.bvx, self.bvy = math.cos(ang)*power, math.sin(ang)*power
        self.bvz = 0.0
        atk.shots += 1
        on = random.random() < (0.40 + (atk.pass_skill()-dfn.dfn_rating())/320.0)
        if on: atk.shots_on += 1
        gk=dfn.gk()
        p_goal = max(0.02, min(0.95, xg * (1.0 - (gk.gk-50)/230.0)))
        tight = abs(shooter.y-gy) < 12 and dist<120
        if tight and random.random()<0.2:
            self.var_timer=1.2; self.var_text="VAR CHECK: GOAL-LINE"
        if random.random() < p_goal:
            atk.goals += 1
            self.phase_reset=True
            self.bvx=self.bvy=self.bvz=0.0
            self.restart=('kickoff', self.pr.x+self.pr.w/2, self.pr.y+self.pr.h/2, dfn);
            self.deadball_timer = max(2.2, STOPPAGE_PAUSE + 0.6)
            self.holder=None; self.gk_hands=False
            self.pass_target=None
            self.goal_flash_timer = 2.4
            self.goal_flash_duration = 2.4
            self.goal_flash_text = f"GOAL! {atk.name}"
            self.goal_flash_color = atk.color
            self.reset_for_kickoff()
            self.place_ball_center()
            self.release_player = None
            self.release_timer = 0.0
            self.add_commentary(f"GOAL! {shooter.name} scores for {atk.name}")
        else:
            if random.random()< (0.55 if not finesse else 0.35):
                self.poss=dfn; self.holder=gk; self.gk_hands=True; self.gk_hold_timer=0.0
                self.bvx=self.bvy=self.bvz=0.0; self.phase_reset=False  # save does NOT reset
                self.pass_target=None
                self.release_player=None
                self.release_timer=0.0
                self.add_commentary(f"{gk.name} makes the save")
            else:
                self.poss=dfn; self.holder=None; self.gk_hands=False
                self.ball[0], self.ball[1] = gk.x + random.uniform(-40,40), gk.y + random.uniform(-30,30)
                self.ball[2]=0.0; self.bvx=random.uniform(-120,120); self.bvy=random.uniform(-100,100); self.bvz=0.0
                # same offside phase (save)
                self.pass_target=None
                self.release_player=None
                self.release_timer=0.0
                self.add_commentary(f"{shooter.name}'s shot is off target")

    # ---------- interceptions ----------
    def try_interception(self, atk:Team, dfn:Team):
        if self.holder is not None or self.restart: return
        if abs(self.bvx)+abs(self.bvy) < 2.0: return
        for d in dfn.outfield():
            vx,vy=self.bvx,self.bvy
            denom=vx*vx+vy*vy
            if denom<1e-6: continue
            t=((d.x-self.ball[0])*vx+(d.y-self.ball[1])*vy)/denom
            if not (0.0<=t<=1.2): continue
            px=self.ball[0]+vx*t; py=self.ball[1]+vy*t
            max_speed=58+46*(d.pac/100.0)
            if math.hypot(px-d.x, py-d.y) <= max(18.0, max_speed*t)+6:
                self.poss=dfn; self.holder=d; self.ball[0],self.ball[1]=px,py; self.bvx=self.bvy=self.bvz=0.0
                self.phase_reset=True
                self.pass_target=None
                self.release_player=None
                self.release_timer=0.0
                self.add_commentary(f"{d.name} steps in to intercept")
                return

    # ---------- duels/tackles (immediate foul) ----------
    def tackle_phase(self, dt):
        if self.holder is None or self.restart: return
        holder=self.holder
        atk = self.H if holder in self.H.players else self.A
        dfn = self.A if atk is self.H else self.H
        defd = min(dfn.outfield() or [dfn.gk()], key=lambda p:(p.x-holder.x)**2+(p.y-holder.y)**2, default=None)
        if not defd: return
        dist = math.hypot(defd.x-holder.x, defd.y-holder.y)
        if dist > TACKLE_RADIUS_PX: return
        if random.random() < dt*2.5:
            duel = (defd.o_dfn()+0.3*defd.pac+random.uniform(-8,8)) - (holder.o_atk()+0.2*holder.pas+random.uniform(-8,8))
            foul_chance = FOUL_BASE_RATE * (1.1 - defd.dis/120.0) * (1.0 + 0.15*(1 if self.minute>=70 else 0))
            if duel > 5:
                self.poss=dfn; self.holder=defd; self.bvx=self.bvy=self.bvz=0.0; self.phase_reset=True
                self.pass_target=None
                self.release_player=None
                self.release_timer=0.0
                self.add_commentary(f"{defd.name} wins it off {holder.name}")
            else:
                if random.random() < foul_chance:
                    self.holder=None
                    self.release_player=None
                    self.release_timer=0.0
                    self.whistle_foul(atk, dfn, holder.x, holder.y, defd)
                    self.add_commentary(f"Foul by {defd.name} on {holder.name}")
                else:
                    self.holder=None; self.bvx=self.bvy=self.bvz=0.0
                    self.pass_target=None
                    self.release_player=None
                    self.release_timer=0.0
                    self.add_commentary(f"Loose ball after the challenge between {holder.name} and {defd.name}")
            if duel < -10 and random.random()<0.22:
                holder.injured=True; holder.injury_timer=30.0; self.injuries_to_process.append(holder)

    # ---------- restarts (set pieces) ----------
    def tick_restart(self, dt):
        if not self.restart: return
        self.deadball_timer = max(0.0, self.deadball_timer - dt)
        if self.deadball_timer > 0.0: return
        kind = self.restart[0]
        if kind == 'kickoff':
            self.reset_for_kickoff()
            self.place_ball_center()
            self.holder = self.nearest(self.poss)  # conceding team takes
            self.release_player = None
            self.release_timer = 0.0
            self.restart=None
            return

        if kind in ('dfk','ifk','corner','throw','goalkick'):
            if kind=='goalkick':
                team=self.restart[3]
                self.keeper_distribute(team)
                self.restart=None
                return
            _, x, y, team = self.restart[:4]
            taker = self.nearest(team)
            if not taker: self.restart=None; return
            taker.x, taker.y = x, y
            mates=[m for m in team.players if m.on_pitch and m is not taker]
            if not mates: self.restart=None; return

            if kind=='corner':
                pattern=random.choice(["near","far","short"])
                if pattern=="short":
                    target=min(mates, key=lambda m:(m.x-x)**2+(m.y-y)**2)
                    self.pass_to(team, taker, target, power=210, ptype="ground")
                else:
                    self.pass_to(team, taker, Player("BOX", "FW", 99,0,0,0,0,0), power=CORNER_CROSS_POWER, ptype="cross")
                self.restart=None; return

            if kind in ('dfk','ifk'):
                defend = self.A if team is self.H else self.H
                wall = sorted(defend.outfield(), key=lambda d:(d.y-y)**2)[:4]
                goal_x = self.pr.right-18 if team is self.H else self.pr.x+18
                dirx = 1 if team is self.H else -1
                wall_xm = (x - self.pr.x)/self.pr.w * PITCH_LEN_M + dirx*WALL_DISTANCE_M
                wx,_ = self.m2p(wall_xm, PITCH_WID_M/2)
                for w in wall:
                    w.x = 0.7*w.x + 0.3*wx + random.uniform(-6,6)
                direct = (kind=='dfk') and (abs(y - (self.pr.y+self.pr.h/2)) < 80) and ((goal_x - x)*dirx > 100)
                if direct:
                    self.on_kick(team, taker, ptype="ground")
                    self.release_player = taker
                    self.release_timer = 0.35
                    ang=math.atan2((self.pr.y+self.pr.h/2)-y, goal_x-x)
                    self.bvx, self.bvy = math.cos(ang)*320, math.sin(ang)*320
                    self.bvz = 180.0; self.ball[2]=1.0
                else:
                    target=min(mates, key=lambda m:(m.x-x)**2+(m.y-y)**2)
                    self.pass_to(team, taker, target, power=210, ptype="ground")
                self.restart=None; return

            if kind=='throw':
                target=min(mates, key=lambda m:(m.x-x)**2+(m.y-y)**2)
                tx = 0.8*target.x + 0.2*(self.pr.x + self.pr.w/2)
                ty = target.y + random.uniform(-12,12)
                self.on_kick(team, taker, ptype="ground")
                self.release_player = taker
                self.release_timer = 0.35
                ang=math.atan2(ty-y, tx-x)
                self.bvx, self.bvy = math.cos(ang)*THROW_FLICK_POWER, math.sin(ang)*THROW_FLICK_POWER
                self.bvz=0.0
                self.restart=None; return
        self.restart=None

    # ---------- out of play handling ----------
    def handle_out(self):
        if self.restart: return
        x,y=self.ball[0], self.ball[1]
        left,right,top,bottom=self.pr.x,self.pr.right,self.pr.y,self.pr.bottom
        if not (x<left or x>right or y<top or y>bottom): return
        self.bvx=self.bvy=self.bvz=0.0; self.holder=None; self.deadball_timer=STOPPAGE_PAUSE
        self.pass_target=None
        self.release_player=None
        self.release_timer=0.0
        atk=self.last_touch_team or self.poss; dfn=self.A if atk is self.H else self.H
        midy=self.pr.y+self.pr.h/2
        if x<left or x>right:
            is_def_last = (self.last_touch_team is dfn)
            if is_def_last:
                cx = left if x<left else right
                cy = top if y < midy else bottom
                self.ball[0]=cx+(12 if x<left else -12); self.ball[1]=cy+(-12 if y<midy else 12)
                self.restart=('corner', self.ball[0], self.ball[1], atk)
                self.add_commentary(f"Corner kick for {atk.name}")
            else:
                gx = left+32 if x<left else right-32
                gy = midy
                self.ball[0],self.ball[1]=gx,gy; self.restart=('goalkick', gx, gy, dfn)
                self.add_commentary(f"Goal kick for {dfn.name}")
        else:
            team = dfn if self.last_touch_team is atk else atk
            tx=clamp(x,left+20,right-20); ty=top+18 if y<top else bottom-18
            self.ball[0],self.ball[1]=tx,ty; self.restart=('throw', tx, ty, team)
            self.add_commentary(f"Throw-in to {team.name}")
    # ---------- one match-minute brain tick ----------
    def step_minute(self):
        # stoppage time rolls
        if self.half == 1 and self.minute == 45:
            self.stoppage = random.randint(1, 3)
        if self.half == 2 and self.minute == 90:
            self.stoppage = random.randint(2, 5)

        # managers
        self.aiH.decide(self.minute, self.H.goals - self.A.goals, self.H.xg, self.A.xg, self.injuries_to_process)
        self.aiA.decide(self.minute, self.A.goals - self.H.goals, self.A.xg, self.H.xg, self.injuries_to_process)
        self.injuries_to_process = [p for p in self.injuries_to_process if p.on_pitch and p.injured]

        # fitness tick
        self.H.fatigue_tick()
        self.A.fatigue_tick()

        # GK 6-second rule (only if actually holding in hands)
        if (self.holder is not None and
            self.holder is self.keeper(self.H if self.holder in self.H.players else self.A) and
            self.gk_hands):
            self.gk_hold_timer += 1.05  # ~1 sec per sim-minute
            if self.gk_hold_timer >= GK_HOLD_MAX_SECONDS:
                team = self.H if self.holder in self.H.players else self.A
                opp  = self.A if team is self.H else self.H
                # IFK for opponent + likely yellow for time-wasting
                self.indirect_free_kick(opp, self.holder.x, self.holder.y, reason="GK 6s")
                if not self.holder.yellow and random.random() < 0.7:
                    self.holder.yellow = True
                    team.yellows += 1

        # no tactical decisions while restart pending
        if self.restart:
            return

        # injury timers tick down
        for p in (self.H.players + self.A.players):
            if p.injured:
                p.injury_timer = max(0.0, p.injury_timer - 1.05)
                if p.injury_timer <= 0.0:
                    p.injured = False

        # if no holder, try to capture
        if self.holder is None:
            self.capture_free_ball()
            return

        # holder decision: pass or shoot, else keep carrying
        atk = self.H if self.holder in self.H.players else self.A
        dfn = self.A if atk is self.H else self.H
        holder = self.holder

        mates = [m for m in atk.players if m.on_pitch and m is not holder]
        lateral_relief = False

        def pass_score(m):
            dx, dy = m.x - holder.x, m.y - holder.y
            dist = math.hypot(dx, dy)
            if dist < 18:
                return -1e9
            goal_x = self.pr.right - 18 if atk is self.H else self.pr.x + 18
            lead = 30 if (atk.tactic in ('attacking', 'press') or m.role in ("winger", "target")) else 18
            tx = m.x + (1 if atk is self.H else -1) * lead
            ty = m.y + clamp((goal_x - m.x), -60, 60) * 0.03

            # congestion along pass corridor
            space = 1.0
            px, py = holder.x, holder.y
            qx, qy = tx, ty
            vx, vy = qx - px, qy - py
            if not (vx == 0 and vy == 0):
                for d in dfn.players:
                    if not d.on_pitch:
                        continue
                    t = max(0, min(1, ((d.x - px) * vx + (d.y - py) * vy) / (vx * vx + vy * vy)))
                    cx, cy = px + vx * t, py + vy * t
                    if math.hypot(d.x - cx, d.y - cy) < 26:
                        space -= 0.28

            forward = (tx - holder.x) if atk is self.H else (holder.x - tx)
            angle_pen = abs(math.atan2(dy, dx)) * 1.4
            width = abs(m.y - (self.pr.y + self.pr.h / 2))
            width_bias = 6.0 if (m.role == "winger" and width > self.pr.h * 0.18) else 0.0
            role_bias = 6.0 if ((holder.role == "bpd" and m.pos != "FW") or (m.role == "target")) else 0.0
            return space * 40 + forward * 0.26 - dist * 0.11 - angle_pen + role_bias + width_bias + random.uniform(-3.5, 3.5)

        took_action = False
        if mates:
            scored = [(pass_score(m), m) for m in mates]
            best_score, cand = max(scored, key=lambda t: t[0])
            goal_x = self.pr.right - 18 if atk is self.H else self.pr.x + 18
            goal_y = self.pr.y + self.pr.h / 2
            dist_goal = math.hypot(holder.x - goal_x, holder.y - goal_y)
            base_appetite = 0.28 + holder.pas / 160.0 + atk.pass_skill() / 360.0
            if dist_goal < 150:
                base_appetite -= 0.08
            pass_appetite = clamp(base_appetite, 0.25, 0.85)
            lateral_relief = best_score > 6.0 or (best_score > -4.0 and random.random() < pass_appetite)
            if lateral_relief and not self.is_offside(atk, cand):
                ptype = "ground"
                flank_zone = abs(cand.y - (self.pr.y + self.pr.h / 2)) > self.pr.h * 0.22
                advanced = (cand.x > holder.x + 26) if atk is self.H else (cand.x < holder.x - 26)
                if cand.role in ("winger", "target") and flank_zone and advanced:
                    ptype = "cross" if random.random() < 0.4 else "chip"
                elif cand.role == "am" and random.random() < 0.25:
                    ptype = "chip"
                self.pass_to(atk, holder, cand, power=320 + random.uniform(-30, 30), ptype=ptype)
                took_action = True

        if not took_action:
            score_diff = (self.H.goals - self.A.goals) if atk is self.H else (self.A.goals - self.H.goals)
            losing = score_diff < 0
            shoot_bias = 0.18 + holder.atk / 250.0 + (atk.atk_rating() - dfn.dfn_rating()) / 320.0
            if losing:
                shoot_bias += 0.14
            else:
                shoot_bias -= 0.05
            if atk is self.A:
                shoot_bias += 0.05
            if self.minute >= 70:
                shoot_bias += 0.04
            if self.minute <= 20 and score_diff > 0:
                shoot_bias -= 0.05
            if losing and self.minute >= 85:
                shoot_bias += 0.10
            if (not losing) and self.minute >= 85:
                shoot_bias -= 0.04
            gx = self.pr.right - 18 if atk is self.H else self.pr.x + 18
            gy = self.pr.y + self.pr.h / 2
            dist_goal = math.hypot(holder.x - gx, holder.y - gy)
            in_good_area = dist_goal < 170 or (dist_goal < 210 and random.random() < 0.5)
            if shoot_bias > 0.32 and dist_goal < 170:
                lateral_relief = False
            if in_good_area and random.random() < max(0.07, shoot_bias):
                self.try_shot(atk, dfn, holder)

        # else keep carrying; tackles may happen in frame_update

    # ---------- per-frame movement/physics ----------
    def frame_update(self, dt):
        self.in_stoppage = bool(self.restart is not None and self.deadball_timer > 0.0)
        in_stoppage = self.in_stoppage

        if not self.pr:
            return

        self.release_timer = max(0.0, self.release_timer - dt)
        if self.goal_flash_timer > 0.0:
            self.goal_flash_timer = max(0.0, self.goal_flash_timer - dt)

        # --- team shape steering ---
        directives = {}
        ball_point = (self.ball[0], self.ball[1])
        if not in_stoppage:
            if self.holder:
                poss_team = self.H if self.holder in self.H.players else self.A
                defend_team = self.A if poss_team is self.H else self.H
                target_point = (self.holder.x, self.holder.y)
                defenders = [p for p in defend_team.players if p.on_pitch and not p.injured and not p.red]
                defenders = sorted(defenders, key=lambda p: (p.x-target_point[0])**2 + (p.y-target_point[1])**2)[:2]
                for d in defenders:
                    directives[d] = target_point
                if self.pass_target and self.pass_target.on_pitch:
                    run_x = self.ball[0] + self.bvx * 0.25
                    run_y = self.ball[1] + self.bvy * 0.25
                    directives[self.pass_target] = (run_x, run_y)
                support = [p for p in poss_team.players if p.on_pitch and p is not self.holder and p is not self.pass_target]
                support = sorted(support, key=lambda p: (p.x-target_point[0])**2 + (p.y-target_point[1])**2)[:2]
                for s in support:
                    sx = 0.7 * target_point[0] + 0.3 * self.ball[0]
                    sy = 0.7 * target_point[1] + 0.3 * self.ball[1] + random.uniform(-6, 6)
                    directives.setdefault(s, (sx, sy))
            else:
                for team in (self.H, self.A):
                    chasers = [p for p in team.players if p.on_pitch and not p.injured and not p.red]
                    chasers = sorted(chasers, key=lambda p: (p.x-ball_point[0])**2 + (p.y-ball_point[1])**2)[:2]
                    for c in chasers:
                        directives[c] = ball_point
        elif self.holder and not self.gk_hands:
            directives[self.holder] = (self.holder.x, self.holder.y)

        def side(team: Team, left_to_right: bool):
            form_fn = anchors_for(team.formation, left_to_right)
            return [self.m2p(*xy) for xy in form_fn]

        anchors_H = side(self.H, True)
        anchors_A = side(self.A, False)

        def steer(team: Team, anchors, dt_local, orders):
            for i, p in enumerate(team.players):
                if not p.on_pitch or p.injured or p.red:
                    continue
                if p in orders:
                    tx, ty = orders[p]
                elif p is self.holder and not self.gk_hands:
                    # slight goal-biased carry
                    goal_x = self.pr.right - 18 if team is self.H else self.pr.x + 18
                    tx = 0.65 * goal_x + 0.35 * p.x
                    ty = 0.82 * (self.pr.y + self.pr.h / 2) + 0.18 * p.y
                else:
                    tx, ty = anchors[i]

                max_speed = 58 + 46 * (p.pac / 100.0)
                stick = {"defensive": 0.75, "balanced": 0.90, "attacking": 1.00, "press": 1.05}[team.tactic]
                dx, dy = (tx - p.x) * 0.8, (ty - p.y) * 0.8
                d = max(1e-5, math.hypot(dx, dy))
                p.vx = (dx / d) * max_speed * stick * dt_local
                p.vy = (dy / d) * max_speed * stick * dt_local
                if p is self.holder and not self.gk_hands:
                    p.vx *= 1.1
                    p.vy *= 1.1
                p.x += p.vx
                p.y += p.vy

        steer(self.H, anchors_H, dt, directives)
        steer(self.A, anchors_A, dt, directives)

        if in_stoppage:
            self.update_bench_positions()
            return

        # --- keeper behaviours ---
        self.keeper_positioning(self.H, True, dt)
        self.keeper_positioning(self.A, False, dt)
        # sweep by defending keeper when danger comes
        defending = self.A if (self.poss is self.H) else self.H
        left_to_right_def = (defending is self.H)
        self.keeper_sweep_try(defending, left_to_right_def, dt)

        # --- collisions (simple) ---
        allp = [p for T in (self.H, self.A) for p in T.players if p.on_pitch and not p.red]
        for i in range(len(allp)):
            for j in range(i + 1, len(allp)):
                a, b = allp[i], allp[j]
                dx, dy = b.x - a.x, b.y - a.y
                dist = math.hypot(dx, dy)
                if dist < max(1e-5, COLLISION_RADIUS_PX):
                    overlap = COLLISION_RADIUS_PX - dist
                    nx, ny = (dx / (dist + 1e-5), dy / (dist + 1e-5))
                    a.x -= nx * (overlap * 0.5)
                    a.y -= ny * (overlap * 0.5)
                    b.x += nx * (overlap * 0.5)
                    b.y += ny * (overlap * 0.5)

        self.update_bench_positions()

        # --- referee trailing play ---
        ref_speed = 120.0 * dt
        rx, ry = self.referee["x"], self.referee["y"]
        bx, by = self.ball[0], self.ball[1]
        cx, cy = self.pr.x + self.pr.w / 2, self.pr.y + self.pr.h / 2
        dirx, diry = bx - cx, by - cy
        dist_dir = math.hypot(dirx, diry)
        trail = 48.0
        if dist_dir > 1e-5:
            tx = bx - (dirx / dist_dir) * trail
            ty = by - (diry / dist_dir) * trail
        else:
            tx, ty = bx, by + trail
        tx = clamp(tx, self.pr.x + 24, self.pr.right - 24)
        ty = clamp(ty, self.pr.y + 24, self.pr.bottom - 24)
        dx, dy = tx - rx, ty - ry
        dist = math.hypot(dx, dy)
        if dist > 1e-3:
            self.referee["x"] += (dx / dist) * ref_speed
            self.referee["y"] += (dy / dist) * ref_speed

        # --- ball physics ---
        if self.holder is not None:
            hx, hy = self.holder.x, self.holder.y
            if self.gk_hands:
                # fully controlled
                self.ball[0], self.ball[1], self.ball[2] = hx, hy, 0.0
                self.bvx = self.bvy = self.bvz = 0.0
            else:
                self.dribble_phase = (self.dribble_phase + dt * 9.0) % (math.pi * 2)
                vx, vy = self.holder.vx, self.holder.vy
                if abs(vx) + abs(vy) < 1e-3:
                    goal_x = self.pr.right - 18 if (self.holder in self.H.players) else self.pr.x + 18
                    ang = math.atan2((self.pr.y + self.pr.h / 2) - hy, goal_x - hx)
                else:
                    ang = math.atan2(vy, vx)
                touch = max(0.0, math.sin(self.dribble_phase)) * DRIBBLE_WOBBLE
                carry = CARRY_OFFSET + touch
                self.ball[0] = hx + math.cos(ang) * carry
                self.ball[1] = hy + math.sin(ang) * carry
                self.ball[2] = 0.0
                self.bvx = self.bvy = self.bvz = 0.0
        else:
            # free ball
            if self.ball[2] > 0.0:  # airborne
                self.bvz -= GRAVITY * dt
                self.ball[2] += self.bvz * dt
                if self.ball[2] <= 0.0:
                    self.ball[2] = 0.0
                    self.bvz = 0.0
                    self.bvx *= BOUNCE_DAMP
                    self.bvy *= BOUNCE_DAMP
            else:
                self.bvx *= BALL_FRICTION_GROUND
                self.bvy *= BALL_FRICTION_GROUND

            self.ball[0] += self.bvx * dt
            self.ball[1] += self.bvy * dt

        # loose ball control should be resolved immediately rather than waiting for the
        # next simulated minute tick; otherwise the closest player appears to "teleport"
        # the ball on the following update when capture_free_ball finally runs.
        if self.holder is None and not self.restart:
            self.capture_free_ball()

        # --- interceptions / tackles / out ---
        atk_for_path = self.last_touch_team or self.poss or self.H
        dfn_for_path = self.A if atk_for_path is self.H else self.H
        self.try_interception(atk_for_path, dfn_for_path)
        self.tackle_phase(dt)
        self.handle_out()

        # --- VAR overlay decay ---
        if self.var_timer > 0.0:
            self.var_timer = max(0.0, self.var_timer - dt)
# =================== Part 4: drawing + main loop ===================

def draw_pitch(surf, rect):
    # grass
    pygame.draw.rect(surf, (30, 120, 30), rect)
    # halfway line
    pygame.draw.line(surf, (240, 240, 240),
                     (rect.x + rect.w // 2, rect.y),
                     (rect.x + rect.w // 2, rect.bottom), 2)
    # center circle + spot
    cc_r = int(CIRCLE_R * (rect.w / PITCH_LEN_M))
    pygame.draw.circle(surf, (240, 240, 240),
                       (rect.x + rect.w // 2, rect.y + rect.h // 2), cc_r, 2)
    pygame.draw.circle(surf, (240, 240, 240),
                       (rect.x + rect.w // 2, rect.y + rect.h // 2), 2)

    # penalty areas (18-yard) and 6-yard boxes (rough visuals)
    def box(x_side):
        w18 = int(BOX_18_M * (rect.w / PITCH_LEN_M))
        h18 = int(40 * (rect.h / PITCH_WID_M))
        w6  = int(BOX_6_M  * (rect.w / PITCH_LEN_M))
        h6  = int(22 * (rect.h / PITCH_WID_M))
        if x_side == "L":
            x18 = rect.x
            x6  = rect.x
        else:
            x18 = rect.right - w18
            x6  = rect.right - w6
        y18 = rect.y + rect.h // 2 - h18 // 2
        y6  = rect.y + rect.h // 2 - h6  // 2
        pygame.draw.rect(surf, (240, 240, 240), pygame.Rect(x18, y18, w18, h18), 2)
        pygame.draw.rect(surf, (240, 240, 240), pygame.Rect(x6, y6, w6, h6), 2)
        # penalty spot
        spot_x = x18 + (w18 if x_side == "L" else 0) - int((w18 if x_side == "R" else -w18) * 0.33)
        pygame.draw.circle(surf, (240, 240, 240),
                           (int(spot_x), rect.y + rect.h // 2), 2)

    box("L")
    box("R")

    # goals (posts + crossbar)
    goal_span = int(7.32 * (rect.h / PITCH_WID_M))
    post_color = (245, 245, 245)
    post_thick = 3
    top_y = rect.y + rect.h // 2 - goal_span // 2
    goal_height = goal_span
    ext = 12
    left_goal = pygame.Rect(rect.x - ext, top_y, ext, goal_height)
    right_goal = pygame.Rect(rect.right, top_y, ext, goal_height)
    pygame.draw.rect(surf, post_color, left_goal, post_thick)
    pygame.draw.rect(surf, post_color, right_goal, post_thick)


def draw_teams(screen, match: 'Match', font, small_font):
    def draw_player(p: 'Player', color):
        r = 9 if p.pos != "GK" else 11
        # outline
        pygame.draw.circle(screen, (0, 0, 0), (int(p.x), int(p.y)), r + 2)
        # body
        pygame.draw.circle(screen, color, (int(p.x), int(p.y)), r)
        # cards
        if p.yellow:
            pygame.draw.circle(screen, (240, 210, 0), (int(p.x) + r + 3, int(p.y) - r - 3), 5)
        if p.red:
            pygame.draw.circle(screen, (200, 40, 40), (int(p.x) - r - 3, int(p.y) - r - 3), 5)
        # number
        num = font.render(str(p.number), True, (255, 255, 255))
        screen.blit(num, (p.x - num.get_width() // 2, p.y - num.get_height() // 2))

    for p in match.H.players:
        if p.on_pitch:
            draw_player(p, match.H.color)
    for p in match.A.players:
        if p.on_pitch:
            draw_player(p, match.A.color)

    # referee
    pygame.draw.circle(screen, (250, 200, 40), (int(match.referee["x"]), int(match.referee["y"])), 8)

    def draw_bench(team: Team, color):
        benchers = [p for p in team.subs if not p.on_pitch]
        benchers += [p for p in team.benched if not p.on_pitch]
        for p in benchers:
            pygame.draw.circle(screen, (0, 0, 0), (int(p.x), int(p.y)), 7)
            pygame.draw.circle(screen, color, (int(p.x), int(p.y)), 6)
            num = small_font.render(str(p.number), True, (255, 255, 255))
            screen.blit(num, (p.x - num.get_width() // 2, p.y - num.get_height() // 2))

    draw_bench(match.H, match.H.alt)
    draw_bench(match.A, match.A.alt)

    # managers
    for team in (match.H, match.A):
        key = id(team)
        if key in match.manager_positions:
            mx, my = match.manager_positions[key]
            rect = pygame.Rect(0, 0, 36, 24)
            rect.center = (int(mx), int(my))
            pygame.draw.rect(screen, (70, 70, 70), rect)
            label = small_font.render(team.manager, True, (255, 255, 255))
            screen.blit(label, (rect.centerx - label.get_width() // 2, rect.bottom + 4))

    # ball (z lifts it slightly)
    x, y, z = match.ball
    pygame.draw.circle(screen, (250, 250, 250), (int(x), int(y - z * 0.25)), 5)

    if match.goal_flash_timer > 0.0 and match.goal_flash_duration > 0.0 and match.goal_flash_text:
        fade = match.goal_flash_timer / match.goal_flash_duration
        overlay = pygame.Surface(screen.get_size(), pygame.SRCALPHA)
        alpha = int(140 * min(1.0, fade))
        color = match.goal_flash_color
        overlay.fill((color[0], color[1], color[2], alpha))
        screen.blit(overlay, (0, 0))
        banner = font.render(match.goal_flash_text, True, (255, 255, 255))
        screen.blit(
            banner,
            (screen.get_width() // 2 - banner.get_width() // 2,
             screen.get_height() // 2 - banner.get_height() // 2)
        )


def draw_hud(screen, match: 'Match', font, speed_idx, small_font):
    y0 = 8
    score = f"{match.H.name} {match.H.goals} - {match.A.goals} {match.A.name}"
    tstr = f"{match.minute:02d}'  H{match.half} +{match.stoppage}"
    spd  = f"Speed x{speed_idx+1}  H:{match.H.tactic}  A:{match.A.tactic}"
    screen.blit(font.render(score, True, (255, 255, 255)), (10, y0))
    screen.blit(font.render(tstr,  True, (255, 255, 255)), (10, y0 + 22))
    screen.blit(font.render(spd,   True, (255, 255, 255)), (10, y0 + 44))
    if match.var_timer > 0.0 and match.var_text:
        lab = font.render(match.var_text, True, (255, 240, 120))
        screen.blit(lab, (screen.get_width() // 2 - lab.get_width() // 2, y0))

    # commentary box
    box_h = 140
    box_rect = pygame.Rect(10, screen.get_height() - box_h - 10, screen.get_width() - 20, box_h)
    pygame.draw.rect(screen, (16, 60, 16), box_rect)
    pygame.draw.rect(screen, (220, 220, 220), box_rect, 2)
    lines = list(reversed(match.commentary[-match.commentary_limit:]))
    for i, line in enumerate(lines):
        text = small_font.render(line, True, (245, 245, 245))
        screen.blit(text, (box_rect.x + 10, box_rect.y + 10 + i * small_font.get_linesize()))


def run_game():
    pygame.init()
    W, H = 1200, 700
    screen = pygame.display.set_mode((W, H))
    pygame.display.set_caption("footy_pro_v6")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont(None, 22)
    small_font = pygame.font.SysFont(None, 18)

    # pitch rect (leave HUD space)
    margin = 40
    pr = pygame.Rect(margin, margin + 40, W - 2 * margin, H - 2 * margin - 60)

    # teams & match
    Hteam = make_team("Home", seed=1, color=(20, 90, 230), alt=(180, 210, 255))
    Ateam = make_team("Away", seed=2, color=(230, 50, 50), alt=(255, 210, 210))
    match = Match(Hteam, Ateam, seed=777)
    match.kickoff(pr)

    # sim-time control
    speed_idx = 1  # 0..2
    sec_per_ingame_min = SIM_SPEEDS[speed_idx]
    ingame_accum = 0.0

    running = True
    while running:
        dt_real = clock.tick(60) / 1000.0

        # input
        for e in pygame.event.get():
            if e.type == pygame.QUIT:
                running = False
            elif e.type == pygame.KEYDOWN:
                if e.key == pygame.K_q:
                    running = False
                elif e.key == pygame.K_1:
                    speed_idx = 0
                elif e.key == pygame.K_2:
                    speed_idx = 1
                elif e.key == pygame.K_3:
                    speed_idx = 2
                elif e.key == pygame.K_SPACE and match.holder and match.gk_hands:
                    # keeper quick throw/roll
                    team = match.H if match.holder in match.H.players else match.A
                    match.keeper_distribute(team)

        # update timing
        sec_per_ingame_min = SIM_SPEEDS[speed_idx]
        ingame_accum += dt_real

        # dead-ball / physics per frame
        match.tick_restart(dt_real)
        match.frame_update(dt_real)

        # advance one "match minute" when enough real time has elapsed
        if ingame_accum >= sec_per_ingame_min:
            ingame_accum -= sec_per_ingame_min
            match.minute += 1

            # half-time / full-time using stoppage time set by step_minute()
            if match.half == 1 and match.minute >= 45 + match.stoppage:
                match.half = 2
                match.minute = 45
                match.stoppage = 0
                match.add_commentary("Half-time whistle")
                # second-half kickoff: give ball to the team that didn't start
                match.poss = match.A if match.poss is match.H else match.H
                match.restart = ('kickoff', pr.x + pr.w / 2, pr.y + pr.h / 2, match.poss)
                match.deadball_timer = 1.2
                match.reset_for_kickoff()
                match.place_ball_center()
                match.release_player = None
                match.release_timer = 0.0
            elif match.half == 2 and match.minute >= 90 + match.stoppage:
                match.add_commentary("Full-time whistle")
                running = False
            else:
                match.step_minute()

        # render
        screen.fill((12, 40, 12))
        draw_pitch(screen, pr)
        draw_teams(screen, match, font, small_font)
        draw_hud(screen, match, font, speed_idx, small_font)
        pygame.display.flip()

    pygame.quit()


if __name__ == "__main__":
    run_game()

