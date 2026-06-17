import csv
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import PillowWriter, FuncAnimation


def rx(t):
    c,s=np.cos(t),np.sin(t)
    return np.array([[1,0,0],[0,c,-s],[0,s,c]])

def ry(t):
    c,s=np.cos(t),np.sin(t)
    return np.array([[c,0,s],[0,1,0],[-s,0,c]])

def rz(t):
    c,s=np.cos(t),np.sin(t)
    return np.array([[c,-s,0],[s,c,0],[0,0,1]])

def rot(a,t):
    if a==0:
        return rx(t)
    if a==1:
        return ry(t)
    return rz(t)

def fk(th,l,ax):
    p=[np.zeros(3)]
    r=np.eye(3)
    rs=[r.copy()]

    for i in range(len(th)):
        r=r@rot(ax[i],th[i])
        p.append(p[-1]+r@np.array([l[i],0,0]))
        rs.append(r.copy())

    return np.array(p),rs

def seg(a,b,c):
    ab=b-a
    v=np.dot(ab,ab)

    if v<1e-12:
        return a,0.0

    t=np.dot(c-a,ab)/v
    t=np.clip(t,0.0,1.0)

    return a+t*ab,t

def clearance(p,obs,rl):
    mn=1e18
    info=None

    if len(obs)==0:
        return mn,info

    for i in range(1,len(p)):
        for k,(c,r) in enumerate(obs):
            q,t=seg(p[i-1],p[i],c)
            d=np.linalg.norm(q-c)-r-rl

            if d<mn:
                mn=d
                info=(i,k,q,t)

    return mn,info

def u(th,l,ax,goal,obs,rl,katt,krep,rho,epsd):
    p,_=fk(th,l,ax)
    val=0.5*katt*np.linalg.norm(p[-1]-goal)**2

    for i in range(1,len(p)):
        for c,r in obs:
            q,t=seg(p[i-1],p[i],c)
            d=np.linalg.norm(q-c)-r-rl
            dd=max(d,epsd)

            if d<=0:
                val+=1e3+1e4*(-d)**2+0.5*krep*(1/epsd-1/rho)**2
            elif d<=rho:
                val+=0.5*krep*(1/dd-1/rho)**2

    return val

def grad(th,l,ax,goal,obs,rl,katt,krep,rho,epsd):
    g=np.zeros_like(th)
    e=1e-5

    for i in range(len(th)):
        a=th.copy()
        b=th.copy()
        a[i]+=e
        b[i]-=e
        g[i]=(u(a,l,ax,goal,obs,rl,katt,krep,rho,epsd)-u(b,l,ax,goal,obs,rl,katt,krep,rho,epsd))/(2*e)

    return g

def f(th,l,ax,goal,obs,rl,katt,krep,rho,epsd,vmax):
    v=-grad(th,l,ax,goal,obs,rl,katt,krep,rho,epsd)
    n=np.linalg.norm(v)

    if n>vmax:
        v=v/n*vmax

    return v

def rk4(th,h,l,ax,goal,obs,rl,katt,krep,rho,epsd,vmax):
    k1=f(th,l,ax,goal,obs,rl,katt,krep,rho,epsd,vmax)
    k2=f(th+h*k1/2,l,ax,goal,obs,rl,katt,krep,rho,epsd,vmax)
    k3=f(th+h*k2/2,l,ax,goal,obs,rl,katt,krep,rho,epsd,vmax)
    k4=f(th+h*k3,l,ax,goal,obs,rl,katt,krep,rho,epsd,vmax)

    return th+h*(k1+2*k2+2*k3+k4)/6

def wrap(th):
    return (th+np.pi)%(2*np.pi)-np.pi

def point_obs_clear(x,obs,m):
    for c,r in obs:
        if np.linalg.norm(x-c)<=r+m:
            return False

    return True

def obs_overlap(c,r,obs,m):
    for c2,r2 in obs:
        if np.linalg.norm(c-c2)<=r+r2+m:
            return True

    return False

def lengths_for_n(n):
    if n==2:
        return np.array([1.0,0.85])
    if n==3:
        return np.array([1.0,0.8,0.6])
    if n==4:
        return np.array([1.0,0.8,0.6,0.45])

    return np.array([1.0,0.85,0.7,0.55,0.4])

def axes_for_n(n):
    base=[2,1]
    return np.array([base[i%2] for i in range(n)])

def path_clearance(th0,tg,l,ax,obs,rl,m=45):
    mn=1e18

    for a in np.linspace(0,1,m):
        th=wrap((1-a)*th0+a*tg)
        p,_=fk(th,l,ax)
        c,_=clearance(p,obs,rl)
        mn=min(mn,c)

    return mn

def hard_obstacle(p0,goal,reach,rng,rl,obs):
    s=p0[-1]
    g=goal
    v=g-s
    nv=np.linalg.norm(v)

    if nv<1e-12:
        return None

    v=v/nv

    for _ in range(300):
        r=rng.uniform(0.16,0.28)

        tmp=rng.normal(size=3)
        n=tmp-np.dot(tmp,v)*v
        nn=np.linalg.norm(n)

        if nn<1e-12:
            continue

        n=n/nn

        a=rng.uniform(0.25,0.75)
        base=s+a*(g-s)

        d=rng.uniform(r+rl+0.015,r+rl+0.11)
        c=base+n*d

        if np.linalg.norm(c)<r+rl+0.20:
            continue
        if obs_overlap(c,r,obs,0.04):
            continue
        if np.linalg.norm(c)>reach*1.15:
            continue

        return c,r

    return None

def sample_case(nj,no,rng,rl,rho,mode="normal"):
    l=lengths_for_n(nj)
    ax=axes_for_n(nj)
    reach=np.sum(l)

    for _ in range(3000):
        th0=rng.uniform(-np.pi,np.pi,nj)
        p0,_=fk(th0,l,ax)

        tg=rng.uniform(-np.pi,np.pi,nj)
        pg,_=fk(tg,l,ax)
        goal=pg[-1]

        if np.linalg.norm(goal)>reach+1e-9:
            continue
        if np.linalg.norm(goal-p0[-1])<0.45:
            continue

        obs=[]
        ok=True

        for _ in range(no):
            placed=False

            for _ in range(2000):
                if mode=="hard":
                    x=hard_obstacle(p0,goal,reach,rng,rl,obs)

                    if x is None:
                        continue

                    c,r=x

                else:
                    r=rng.uniform(0.12,0.22)

                    c=rng.uniform(
                        low=np.array([-0.25*reach,-0.8*reach,-0.6*reach]),
                        high=np.array([0.95*reach,0.8*reach,0.6*reach])
                    )

                if np.linalg.norm(c)<r+rl+0.25:
                    continue
                if obs_overlap(c,r,obs,0.08):
                    continue
                if clearance(p0,[(c,r)],rl)[0]<=0.08:
                    continue
                if clearance(pg,[(c,r)],rl)[0]<=0.08:
                    continue
                if np.linalg.norm(goal-c)<=r+rl+0.08:
                    continue

                placed=True
                obs.append((c,r))
                break

            if not placed:
                ok=False
                break

        if not ok:
            continue

        c0,_=clearance(p0,obs,rl)
        cg,_=clearance(pg,obs,rl)

        if c0<=0:
            continue
        if cg<=0:
            continue
        if not point_obs_clear(goal,obs,rl+0.06):
            continue
        if max([r for _,r in obs],default=0.0)>=rho:
            continue

        if mode=="hard" and no>0:
            pc=path_clearance(th0,tg,l,ax,obs,rl)

            if pc>0.10:
                continue
            if pc<-0.12:
                continue

        return l,ax,th0,goal,obs

    raise RuntimeError(f"failed to sample case: joints={nj}, obs={no}, mode={mode}")

def simulate(l,ax,th,goal,obs,rl,katt,krep,rho,epsd,h,eps,nmax,vmax):
    ps=[]
    ths=[]
    es=[]
    cs=[]
    us=[]

    ok=False

    p,_=fk(th,l,ax)
    cmin,_=clearance(p,obs,rl)

    if cmin<=0:
        return {
            "ok":False,
            "p":np.array([p]),
            "th":np.array([th.copy()]),
            "e":np.array([np.linalg.norm(p[-1]-goal)]),
            "c":np.array([cmin]),
            "u":np.array([u(th,l,ax,goal,obs,rl,katt,krep,rho,epsd)]),
            "goal":goal,
            "obs":obs,
            "rl":rl
        }

    for _ in range(nmax):
        p,_=fk(th,l,ax)
        cmin,_=clearance(p,obs,rl)
        e=np.linalg.norm(p[-1]-goal)
        uu=u(th,l,ax,goal,obs,rl,katt,krep,rho,epsd)

        ps.append(p.copy())
        ths.append(th.copy())
        es.append(e)
        cs.append(cmin)
        us.append(uu)

        if e<eps and cmin>0:
            ok=True
            break

        hh=h
        moved=False
        best=None

        for _ in range(14):
            nt=wrap(rk4(th,hh,l,ax,goal,obs,rl,katt,krep,rho,epsd,vmax))
            npnt,_=fk(nt,l,ax)
            ne=np.linalg.norm(npnt[-1]-goal)
            nc,_=clearance(npnt,obs,rl)
            nu=u(nt,l,ax,goal,obs,rl,katt,krep,rho,epsd)

            if nc>0 and nu<uu:
                th=nt
                moved=True
                break

            score=(nc>0,uu-nu,e-ne,nc)

            if best is None or score>best[0]:
                best=(score,nt,nu,nc)

            hh*=0.5

        if not moved:
            if best is not None and best[0][0] and (best[2]<uu or best[0][2]>1e-6):
                th=best[1]
            else:
                break

    return {
        "ok":ok,
        "p":np.array(ps),
        "th":np.array(ths),
        "e":np.array(es),
        "c":np.array(cs),
        "u":np.array(us),
        "goal":goal,
        "obs":obs,
        "rl":rl
    }

def sphere_wire(ax,c,r):
    a=np.linspace(0,2*np.pi,28)
    b=np.linspace(0,np.pi,14)

    x=c[0]+r*np.outer(np.cos(a),np.sin(b))
    y=c[1]+r*np.outer(np.sin(a),np.sin(b))
    z=c[2]+r*np.outer(np.ones_like(a),np.cos(b))

    ax.plot_wireframe(x,y,z,linewidth=0.4,alpha=0.4)

def seteq(ax,x,y,z):
    mx=np.array([x.min(),y.min(),z.min()])
    ma=np.array([x.max(),y.max(),z.max()])
    mid=(mx+ma)/2
    r=max((ma-mx).max()/2,0.5)

    ax.set_xlim(mid[0]-r,mid[0]+r)
    ax.set_ylim(mid[1]-r,mid[1]+r)
    ax.set_zlim(mid[2]-r,mid[2]+r)

def xyz_bounds(p,goal,obs):
    xs=[p[:,:,0].ravel(),np.array([goal[0]])]
    ys=[p[:,:,1].ravel(),np.array([goal[1]])]
    zs=[p[:,:,2].ravel(),np.array([goal[2]])]

    for c,r in obs:
        xs.append(np.array([c[0]-r,c[0]+r]))
        ys.append(np.array([c[1]-r,c[1]+r]))
        zs.append(np.array([c[2]-r,c[2]+r]))

    return np.concatenate(xs),np.concatenate(ys),np.concatenate(zs)

def plot_traj3d(res,save_path):
    p=res["p"]
    goal=res["goal"]
    obs=res["obs"]

    if len(p)==0:
        return

    fig=plt.figure(figsize=(8,7))
    ax=fig.add_subplot(111,projection="3d")

    step=max(1,len(p)//25)

    for q in p[::step]:
        ax.plot(q[:,0],q[:,1],q[:,2],alpha=0.18)

    ee=p[:,-1,:]
    ax.plot(ee[:,0],ee[:,1],ee[:,2],linewidth=2.5,label="end-effector path")

    for c,r in obs:
        sphere_wire(ax,c,r)

    ax.scatter(goal[0],goal[1],goal[2],s=70,label="goal")
    ax.scatter(0,0,0,s=50,label="base")
    ax.plot(p[-1,:,0],p[-1,:,1],p[-1,:,2],linewidth=3,label="final pose")

    xs,ys,zs=xyz_bounds(p,goal,obs)
    seteq(ax,xs,ys,zs)

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.set_title("Robot arm trajectory")
    ax.legend()

    plt.tight_layout()
    plt.savefig(save_path,dpi=180)
    plt.close(fig)

def plot_logs(res,save_path):
    e=res["e"]
    c=res["c"]
    uu=res["u"]
    th=res["th"]

    if len(e)==0:
        return

    fig,axs=plt.subplots(2,2,figsize=(10,7))

    axs[0,0].plot(e)
    axs[0,0].set_title("goal error")
    axs[0,0].set_xlabel("step")
    axs[0,0].set_ylabel("error")
    axs[0,0].grid()

    axs[0,1].plot(c)
    axs[0,1].axhline(0,linestyle="--")
    axs[0,1].set_title("minimum clearance")
    axs[0,1].set_xlabel("step")
    axs[0,1].set_ylabel("clearance")
    axs[0,1].grid()

    axs[1,0].plot(uu)
    axs[1,0].set_title("potential")
    axs[1,0].set_xlabel("step")
    axs[1,0].set_ylabel("U")
    axs[1,0].grid()

    for i in range(th.shape[1]):
        axs[1,1].plot(th[:,i],label=f"theta {i+1}")

    axs[1,1].set_title("joint angles")
    axs[1,1].set_xlabel("step")
    axs[1,1].set_ylabel("rad")
    axs[1,1].grid()
    axs[1,1].legend()

    plt.tight_layout()
    plt.savefig(save_path,dpi=180)
    plt.close(fig)

def save_gif(res,save_path):
    p=res["p"]
    goal=res["goal"]
    obs=res["obs"]

    if len(p)<=1:
        return

    idx=np.arange(len(p))

    if len(idx)>120:
        idx=np.linspace(0,len(p)-1,120).astype(int)

    pp=p[idx]
    ee=pp[:,-1,:]

    fig=plt.figure(figsize=(7,6))
    ax=fig.add_subplot(111,projection="3d")

    xs,ys,zs=xyz_bounds(pp,goal,obs)

    def draw_static():
        ax.clear()

        for c,r in obs:
            sphere_wire(ax,c,r)

        ax.scatter(goal[0],goal[1],goal[2],s=70)
        ax.scatter(0,0,0,s=50)

        seteq(ax,xs,ys,zs)

        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")
        ax.set_title("Robot arm motion")

    def upd(i):
        draw_static()

        q=pp[i]

        ax.plot(ee[:i+1,0],ee[:i+1,1],ee[:i+1,2],linewidth=2)
        ax.plot(q[:,0],q[:,1],q[:,2],linewidth=3,marker="o")

        return []

    ani=FuncAnimation(fig,upd,frames=len(pp),interval=90,blit=False)
    ani.save(save_path,writer=PillowWriter(fps=10))

    plt.close(fig)

def metrics(res):
    p=res["p"]
    th=res["th"]

    if len(p)==0 or len(th)==0:
        return {
            "success":0,
            "steps":0,
            "final_error":np.nan,
            "min_clearance":np.nan,
            "final_clearance":np.nan,
            "Ltask":np.nan,
            "Ljoint":np.nan
        }

    ee=p[:,-1,:]

    ltask=0.0
    ljoint=0.0

    if len(ee)>=2:
        ltask=np.sum(np.linalg.norm(np.diff(ee,axis=0),axis=1))

    if len(th)>=2:
        ljoint=np.sum(np.linalg.norm(np.diff(th,axis=0),axis=1))

    return {
        "success":int(res["ok"]),
        "steps":len(res["e"]),
        "final_error":float(res["e"][-1]),
        "min_clearance":float(np.min(res["c"])),
        "final_clearance":float(res["c"][-1]),
        "Ltask":float(ltask),
        "Ljoint":float(ljoint)
    }

def arrstr(a):
    return " ".join(map(lambda x:f"{float(x):.17g}",np.ravel(a)))

def save_trial_csv(d,nj,no,tr,l,ax,th0,goal,obs,res,pars):
    d=Path(d)

    with open(d/"params.csv","w",newline="",encoding="utf-8-sig") as f:
        w=csv.writer(f)
        w.writerow(["key","value"])
        w.writerow(["obs",no])
        w.writerow(["joints",nj])
        w.writerow(["trial",tr])
        w.writerow(["lengths",arrstr(l)])
        w.writerow(["axes",arrstr(ax)])
        w.writerow(["theta0",arrstr(th0)])
        w.writerow(["goal",arrstr(goal)])

        for k,v in pars.items():
            w.writerow([k,v])

    with open(d/"obstacles.csv","w",newline="",encoding="utf-8-sig") as f:
        w=csv.writer(f)
        w.writerow(["k","cx","cy","cz","r"])

        for k,(c,r) in enumerate(obs):
            w.writerow([
                k,
                f"{float(c[0]):.17g}",
                f"{float(c[1]):.17g}",
                f"{float(c[2]):.17g}",
                f"{float(r):.17g}"
            ])

    th=res["th"]
    p=res["p"]
    e=res["e"]
    c=res["c"]
    uu=res["u"]

    with open(d/"states.csv","w",newline="",encoding="utf-8-sig") as f:
        head=["step","error","clearance","potential","ee_x","ee_y","ee_z"]
        head+=[f"theta_{i}" for i in range(th.shape[1])]

        w=csv.writer(f)
        w.writerow(head)

        for t in range(len(th)):
            ee=p[t,-1]
            row=[
                t,
                f"{float(e[t]):.17g}",
                f"{float(c[t]):.17g}",
                f"{float(uu[t]):.17g}",
                f"{float(ee[0]):.17g}",
                f"{float(ee[1]):.17g}",
                f"{float(ee[2]):.17g}",
            ]
            row+=[f"{float(x):.17g}" for x in th[t]]
            w.writerow(row)

    with open(d/"links.csv","w",newline="",encoding="utf-8-sig") as f:
        head=["step"]

        for i in range(p.shape[1]):
            head+=[f"p{i}_x",f"p{i}_y",f"p{i}_z"]

        w=csv.writer(f)
        w.writerow(head)

        for t in range(p.shape[0]):
            row=[t]

            for i in range(p.shape[1]):
                row+=[
                    f"{float(p[t,i,0]):.17g}",
                    f"{float(p[t,i,1]):.17g}",
                    f"{float(p[t,i,2]):.17g}",
                ]

            w.writerow(row)

def read_params(path):
    data={}

    with open(path,newline="",encoding="utf-8-sig") as f:
        r=csv.reader(f)
        next(r)

        for k,v in r:
            data[k]=v

    return data

def farr(s):
    if s.strip()=="":
        return np.array([])

    return np.array(list(map(float,s.split())))

def load_trial_csv(d):
    d=Path(d)
    pms=read_params(d/"params.csv")

    l=farr(pms["lengths"])
    ax=farr(pms["axes"]).astype(int)
    th0=farr(pms["theta0"])
    goal=farr(pms["goal"])

    obs=[]

    with open(d/"obstacles.csv",newline="",encoding="utf-8-sig") as f:
        r=csv.DictReader(f)

        for row in r:
            c=np.array([
                float(row["cx"]),
                float(row["cy"]),
                float(row["cz"])
            ])
            rr=float(row["r"])
            obs.append((c,rr))

    states=[]

    with open(d/"states.csv",newline="",encoding="utf-8-sig") as f:
        r=csv.DictReader(f)

        for row in r:
            states.append(row)

    th=[]
    e=[]
    c=[]
    uu=[]

    nj=len(l)

    for row in states:
        e.append(float(row["error"]))
        c.append(float(row["clearance"]))
        uu.append(float(row["potential"]))
        th.append([float(row[f"theta_{i}"]) for i in range(nj)])

    th=np.array(th)
    e=np.array(e)
    c=np.array(c)
    uu=np.array(uu)

    ps=[]

    with open(d/"links.csv",newline="",encoding="utf-8-sig") as f:
        r=csv.DictReader(f)

        for row in r:
            q=[]

            for i in range(nj+1):
                q.append([
                    float(row[f"p{i}_x"]),
                    float(row[f"p{i}_y"]),
                    float(row[f"p{i}_z"])
                ])

            ps.append(q)

    p=np.array(ps)

    ok=False

    if len(e)>0:
        ok=(e[-1]<float(pms["eps"]) and c[-1]>0)

    return {
        "l":l,
        "ax":ax,
        "th0":th0,
        "goal":goal,
        "obs":obs,
        "rl":float(pms["rl"]),
        "p":p,
        "th":th,
        "e":e,
        "c":c,
        "u":uu,
        "ok":ok
    }

def save_summary_heatmap(mat,xt,yt,title,save_path):
    fig,ax=plt.subplots(figsize=(7,5))
    im=ax.imshow(mat,aspect="auto")

    ax.set_xticks(np.arange(len(xt)))
    ax.set_xticklabels(xt)
    ax.set_yticks(np.arange(len(yt)))
    ax.set_yticklabels(yt)

    ax.set_xlabel("number of joints")
    ax.set_ylabel("number of obstacles")
    ax.set_title(title)

    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            txt="nan" if np.isnan(mat[i,j]) else f"{mat[i,j]:.2f}"
            ax.text(j,i,txt,ha="center",va="center")

    plt.colorbar(im,ax=ax)
    plt.tight_layout()
    plt.savefig(save_path,dpi=180)
    plt.close(fig)

def replay_trial(d):
    tr=load_trial_csv(d)

    res={
        "ok":tr["ok"],
        "p":tr["p"],
        "th":tr["th"],
        "e":tr["e"],
        "c":tr["c"],
        "u":tr["u"],
        "goal":tr["goal"],
        "obs":tr["obs"],
        "rl":tr["rl"]
    }

    plot_traj3d(res,Path(d)/"replay_traj3d.png")
    plot_logs(res,Path(d)/"replay_logs.png")
    save_gif(res,Path(d)/"replay_arm.gif")

def main():
    mode="hard"
    # mode="hard"

    out=Path("figs_hard" if mode=="hard" else "figs")
    out.mkdir(exist_ok=True)
    (out/"summary").mkdir(exist_ok=True)

    seed=0
    rng=np.random.default_rng(seed)

    n_trials=5
    obs_list=[0,1,2,3]
    joint_list=[2,3,4,5]

    rl=0.04
    katt=1.0
    krep=5e-4
    rho=0.6
    epsd=1e-3
    h=0.04
    eps=0.05
    nmax=800
    vmax=3.0

    pars={
        "rl":rl,
        "katt":katt,
        "krep":krep,
        "rho":rho,
        "epsd":epsd,
        "h":h,
        "eps":eps,
        "nmax":nmax,
        "vmax":vmax,
        "seed":seed,
        "n_trials":n_trials,
        "mode":mode
    }

    rows=[]

    for no in obs_list:
        for nj in joint_list:
            for tr in range(1,n_trials+1):
                d=out/f"obs_{no}"/f"joints_{nj}"/f"trial_{tr:02d}"
                d.mkdir(parents=True,exist_ok=True)

                l,ax,th0,goal,obs=sample_case(nj,no,rng,rl,rho,mode)
                res=simulate(l,ax,th0,goal,obs,rl,katt,krep,rho,epsd,h,eps,nmax,vmax)

                save_trial_csv(d,nj,no,tr,l,ax,th0,goal,obs,res,pars)
                plot_traj3d(res,d/"traj3d.png")
                plot_logs(res,d/"logs.png")
                save_gif(res,d/"arm.gif")

                m=metrics(res)

                row={
                    "mode":mode,
                    "obs":no,
                    "joints":nj,
                    "trial":tr,
                    "success":m["success"],
                    "steps":m["steps"],
                    "final_error":m["final_error"],
                    "min_clearance":m["min_clearance"],
                    "final_clearance":m["final_clearance"],
                    "Ltask":m["Ltask"],
                    "Ljoint":m["Ljoint"],
                    "goal_x":goal[0],
                    "goal_y":goal[1],
                    "goal_z":goal[2],
                    "theta0":arrstr(th0),
                    "lengths":arrstr(l),
                    "axes":arrstr(ax)
                }

                for i,(c,r) in enumerate(obs):
                    row[f"obs{i}_x"]=c[0]
                    row[f"obs{i}_y"]=c[1]
                    row[f"obs{i}_z"]=c[2]
                    row[f"obs{i}_r"]=r

                rows.append(row)

                print(
                    f"done mode={mode} obs={no} joints={nj} trial={tr} "
                    f"success={m['success']} err={m['final_error']:.4f} "
                    f"clear={m['min_clearance']:.4f}"
                )

    keys=set()

    for r in rows:
        keys|=set(r.keys())

    base=[
        "mode","obs","joints","trial","success","steps",
        "final_error","min_clearance","final_clearance",
        "Ltask","Ljoint","goal_x","goal_y","goal_z",
        "theta0","lengths","axes"
    ]

    keys=base+sorted([k for k in keys if k not in base])

    with open(out/"summary"/"results.csv","w",newline="",encoding="utf-8-sig") as f:
        w=csv.DictWriter(f,fieldnames=keys)
        w.writeheader()

        for r in rows:
            w.writerow(r)

    sr=np.full((len(obs_list),len(joint_list)),np.nan)
    fe=np.full((len(obs_list),len(joint_list)),np.nan)
    mc=np.full((len(obs_list),len(joint_list)),np.nan)

    for i,no in enumerate(obs_list):
        for j,nj in enumerate(joint_list):
            sub=[r for r in rows if r["obs"]==no and r["joints"]==nj]

            sr[i,j]=np.mean([r["success"] for r in sub])
            fe[i,j]=np.mean([r["final_error"] for r in sub])
            mc[i,j]=np.mean([r["min_clearance"] for r in sub])

    save_summary_heatmap(sr,joint_list,obs_list,"success rate",out/"summary"/"success_rate.png")
    save_summary_heatmap(fe,joint_list,obs_list,"mean final error",out/"summary"/"mean_final_error.png")
    save_summary_heatmap(mc,joint_list,obs_list,"mean min clearance",out/"summary"/"mean_min_clearance.png")

if __name__=="__main__":
    main()