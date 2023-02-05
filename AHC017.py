import io
import sys
import glob

#同じフォルダにtest_caseという名称でテストケースを置けば、test関数等が動く
files = glob.glob("C:\\Users\katonyonko\\OneDrive\\デスクトップ\\AHC017/test_case/*")
_INPUT = ""
for file in files:
  # sample_file.txtファイルを"読み込みモード"で開く
  file_data = open(file, "r")
  _INPUT += file_data.read()
  # 開いたファイルを閉じる
  file_data.close()
  
sys.stdin = io.StringIO(_INPUT)

#以下が提出するコード
import time
import math
from random import randint, shuffle, uniform, choice, sample
from heapq import heappop, heappush
from collections import deque

class Ahc017:

  def __init__(self, start, TL, N, M, D, K, G, E, points, djk_start):
    self.start=start
    self.TL=TL
    self.N = N
    self.M = M
    self.D = D
    self.K = K
    self.G = G
    self.E = E
    self.points = points
    self.djk_start_cand=djk_start
    self.djk_start=[]
    self.ans = [0]*self.M
    self.cnt = [0]*self.D
    self.score=[0]*self.D
    self.changed = 0
    self.updated=0

  def dijkstra(self,no_go,s):
    ans=0
    for i in range(len(s)):
      done=[False]*self.N
      inf=10**9
      C=[inf]*self.N
      C[s[i]]=0
      h=[]
      heappush(h,(0,s[i]))
      while h:
        x,y=heappop(h)
        if done[y]:
          continue
        done[y]=True
        for v in self.G[y]:
          if no_go[v[2]]==1: continue
          if C[v[1]]>C[y]+v[0]:
            C[v[1]]=C[y]+v[0]
            heappush(h,(C[v[1]],v[1]))
      ans+=sum(C)
    return ans/len(s)

  def bfs(self,s):
    inf=10**5
    D=[inf]*self.N
    D[s]=0
    dq=deque()
    dq.append(s)
    while dq:
      x=dq.popleft()
      for w,y,i in self.G[x]:
        if D[y]>D[x]+1:
          D[y]=D[x]+1
          dq.append(y)
    return [(D[i],i) for i in range(self.N)]

  def simple_score(self,ans):
    res=0
    for i in range(self.D):
      no_go=[0 if ans[j]!=i else 1 for j in range(self.M)]
      res+=self.dijkstra(no_go,self.djk_start)
    return res

  def init_ans(self):
    shuffle(self.E)
    cnt = [0]*(self.D*self.N)
    for i in range(self.M):
      w,u,v,k=self.E[i]
      tmp=[(j,cnt[u*self.D+j]+cnt[v*self.D+j]) for j in range(self.D) if self.cnt[j]<self.K]
      m=min([tmp[j][1] for j in range(len(tmp))])
      x=choice([tmp[j][0] for j in range(len(tmp)) if tmp[j][1]==m])
      cnt[u*self.D+x]+=1
      cnt[v*self.D+x]+=1
      self.ans[k]=x
      self.cnt[x]+=1

  def add_start_point(self,s):
    for i in range(len(s)): self.djk_start.append(s[i])
    for i in range(self.D): self.score[i]=(self.score[i]*(len(self.djk_start)-len(s))+self.dijkstra([1 if self.ans[j]==i else 0 for j in range(self.M)],s)*len(s))/len(self.djk_start)

  def init_ans_repeat(self):
    shuffle(self.E)
    cnt = [0]*(self.D*self.N)
    ans_tmp=self.ans.copy()
    cnt_tmp=self.cnt.copy()
    score_tmp=self.score.copy()
    for i in range(self.M):
      w,u,v,k=self.E[i]
      tmp=[(j,cnt[u*self.D+j]+cnt[v*self.D+j]) for j in range(self.D) if self.cnt[j]<self.K]
      m=min([tmp[j][1] for j in range(len(tmp))])
      x=choice([tmp[j][0] for j in range(len(tmp)) if tmp[j][1]==m])
      cnt[u*self.D+x]+=1
      cnt[v*self.D+x]+=1
      ans_tmp[k]=x
      cnt_tmp[x]+=1
    for i in range(self.D): score_tmp[i]=self.dijkstra([1 if self.ans[j]==i else 0 for j in range(self.M)],self.djk_start)
    return ans_tmp, cnt_tmp, score_tmp

  def init_ans2(self):
    D=sorted(self.bfs(self.djk_start[0]))
    D=[D[i][1] for i in range(self.N)]
    used=set()
    id=[]
    for i in range(self.N):
      for w,u,j in self.G[D[i]]:
        if j not in used: id.append(j); used.add(j)
    cnt = [0]*(self.D*self.N)
    for i in range(self.M):
      w,u,v,k=self.E[id[i]]
      tmp=[(j,cnt[u*self.D+j]+cnt[v*self.D+j]) for j in range(self.D) if self.cnt[j]<self.K]
      m=min([tmp[j][1] for j in range(len(tmp))])
      x=choice([tmp[j][0] for j in range(len(tmp)) if tmp[j][1]==m])
      cnt[u*self.D+x]+=1
      cnt[v*self.D+x]+=1
      self.ans[k]=x
      self.cnt[x]+=1
    for i in range(self.D): self.score[i]=self.dijkstra([1 if self.ans[j]==i else 0 for j in range(self.M)],self.djk_start)

  def random_change(self,n):
    '''
    ランダムに２つの日を抽出し、n個の要素を選択して一方からもう一方に工事日を移し、その際のスコア（１点からのダイクストラで計算したもの）の変動を計算する。
    '''
    a=randint(0,self.D-1)
    b=randint(0,self.D-2)
    if b>=a: b+=1
    x=[i for i in range(self.M) if self.ans[i]==a]
    y=[i for i in range(self.M) if self.ans[i]==b]
    n=min(len(x),self.K-len(y),n)
    modify=sample(x,n)
    tmpa,tmpb=[1 if self.ans[i]==a else 0 for i in range(self.M)],[1 if self.ans[i]==b else 0 for i in range(self.M)]
    for i in range(n): tmpa[modify[i]]=0; tmpb[modify[i]]=1
    return modify,a,b,self.dijkstra(tmpa,self.djk_start),self.dijkstra(tmpb,self.djk_start)

  def random_change2(self,p):
    '''
    ランダムに２つの日を抽出し、各辺についてpの確率で交換する。
    '''
    a=randint(0,self.D-1)
    b=randint(0,self.D-2)
    if b>=a: b+=1
    an,bn=self.cnt[a],self.cnt[b]
    modify=[]
    tmpa,tmpb=[1 if self.ans[i]==a else 0 for i in range(self.M)],[1 if self.ans[i]==b else 0 for i in range(self.M)]
    for i in range(self.M):
      if p>uniform(0,1):
        if tmpa[i]==1 and bn<self.K: tmpa[i]=0; tmpb[i]=1; an-=1; bn+=1; modify.append(i)
        elif tmpb[i]==1 and an<self.K: tmpa[i]=1; tmpb[i]=0; an+=1; bn-=1; modify.append(i)
    return modify,a,b,self.dijkstra(tmpa,self.djk_start),self.dijkstra(tmpb,self.djk_start)

  def exchange(self,n):
    '''
    ランダムに２つの日を抽出し、n個の辺を交換する。
    '''
    a=randint(0,self.D-1)
    b=randint(0,self.D-2)
    if b>=a: b+=1
    n=min(n,self.cnt[a],self.cnt[b])
    tmpa,tmpb=[1 if self.ans[i]==a else 0 for i in range(self.M)],[1 if self.ans[i]==b else 0 for i in range(self.M)]
    modify=sample([i for i in range(self.M) if self.ans[i]==a],n)+sample([i for i in range(self.M) if self.ans[i]==b],n)
    for i in range(2*n):
      if tmpa[modify[i]]==1: tmpa[modify[i]]=0; tmpb[modify[i]]=1
      else: tmpa[modify[i]]=1; tmpb[modify[i]]=0
    return modify,a,b,self.dijkstra(tmpa,self.djk_start),self.dijkstra(tmpb,self.djk_start)

  def line_change(self,p):
    '''
    ランダムに２つの日を抽出、さらにグラフを構成する円に交わる適当な直線を引き、その一方にある辺をpの確率で交換する。
    '''
    a=randint(0,self.D-1)
    b=randint(0,self.D-2)
    if b>=a: b+=1
    an,bn=self.cnt[a],self.cnt[b]
    r,theta,theta2=uniform(0,1),uniform(0,1)*2*math.pi,uniform(0,1)*2*math.pi
    x,y=500+500*r*math.cos(theta),500+500*r*math.sin(theta)
    c=math.tan(theta2)
    d=y-x*c
    tmpa,tmpb=[1 if self.ans[i]==a else 0 for i in range(self.M)],[1 if self.ans[i]==b else 0 for i in range(self.M)]
    modifya,modifyb=[],[]
    tmp=[self.points[i][1]-self.points[i][0]*c+d>0 for i in range(self.N)]
    for i in range(self.M):
      w,u,v,k=self.E[i]
      if tmp[u]==True or tmp[v]==True and uniform(0,1)<p:
        if tmpa[k]==1: tmpa[k]=0; tmpb[k]=1; an-=1; bn+=1; modifya.append(k)
        elif tmpb[k]==1: tmpa[k]=1; tmpb[k]=0; an+=1; bn-=1; modifyb.append(k)
    if an>self.K:
      for i in range(an-self.K):
        x=modifyb.pop(); tmpa[x]=0; tmpb[x]=1; an-=1; bn+=1
    if bn>self.K:
      for i in range(bn-self.K):
        x=modifya.pop(); tmpa[x]=1; tmpb[x]=0; an+=1; bn-=1
    return modifya+modifyb,a,b,self.dijkstra(tmpa,self.djk_start),self.dijkstra(tmpb,self.djk_start)

  def line_on_change(self,p):
    '''
    ランダムに２つの日を抽出、さらにグラフを構成する円に交わる適当な直線を引き、その一方にある辺をランダムに交換する。
    '''
    a=randint(0,self.D-1)
    b=randint(0,self.D-2)
    if b>=a: b+=1
    an,bn=self.cnt[a],self.cnt[b]
    r,theta,theta2=uniform(0,1),uniform(0,1)*2*math.pi,uniform(0,1)*2*math.pi
    x,y=500+500*r*math.cos(theta),500+500*r*math.sin(theta)
    c=math.tan(theta2)
    d=y-x*c
    tmpa,tmpb=[1 if self.ans[i]==a else 0 for i in range(self.M)],[1 if self.ans[i]==b else 0 for i in range(self.M)]
    modifya,modifyb=[],[]
    tmp=[self.points[i][1]-self.points[i][0]*c+d>0 for i in range(self.N)]
    for i in range(self.M):
      w,u,v,k=self.E[i]
      if tmp[u]*tmp[v]==False and uniform(0,1)<p:
        if tmpa[k]==1: tmpa[k]=0; tmpb[k]=1; an-=1; bn+=1; modifya.append(k)
        elif tmpb[k]==1: tmpa[k]=1; tmpb[k]=0; an+=1; bn-=1; modifyb.append(k)
    if an>self.K:
      for i in range(an-self.K):
        x=modifyb.pop(); tmpa[x]=0; tmpb[x]=1; an-=1; bn+=1
    if bn>self.K:
      for i in range(bn-self.K):
        x=modifya.pop(); tmpa[x]=1; tmpb[x]=0; an+=1; bn-=1
    return modifya+modifyb,a,b,self.dijkstra(tmpa,self.djk_start),self.dijkstra(tmpb,self.djk_start)

  def circle_change(self,p,maxr=300,minr=30):
    '''
    ランダムに２つの日を抽出、さらにグラフの内側に円を描き、その内側の辺をランダムに交換する（maxr,minrは内側の円の大きさの最大値と最小値）。
    '''
    a=randint(0,self.D-1)
    b=randint(0,self.D-2)
    if b>=a: b+=1
    an,bn=self.cnt[a],self.cnt[b]
    r,theta,r2=uniform(0,1),uniform(0,1)*2*math.pi,minr+uniform(0,1)*(maxr-minr)
    x,y=500+500*r*math.cos(theta),500+500*r*math.sin(theta)
    tmpa,tmpb=[1 if self.ans[i]==a else 0 for i in range(self.M)],[1 if self.ans[i]==b else 0 for i in range(self.M)]
    modifya,modifyb=[],[]
    tmp=[(self.points[i][0]-x)**2+(self.points[i][1]-y)**2<r2**2 for i in range(self.N)]
    for i in range(self.M):
      w,u,v,k=self.E[i]
      if tmp[u]==True or tmp[v]==True and uniform(0,1)<p:
        if tmpa[k]==1: tmpa[k]=0; tmpb[k]=1; an-=1; bn+=1; modifya.append(k)
        elif tmpb[k]==1: tmpa[k]=1; tmpb[k]=0; an+=1; bn-=1; modifyb.append(k)
    if an>self.K:
      for i in range(an-self.K):
        x=modifyb.pop(); tmpa[x]=0; tmpb[x]=1; an-=1; bn+=1
    if bn>self.K:
      for i in range(bn-self.K):
        x=modifya.pop(); tmpa[x]=1; tmpb[x]=0; an+=1; bn-=1
    return modifya+modifyb,a,b,self.dijkstra(tmpa,self.djk_start),self.dijkstra(tmpb,self.djk_start)
    
  def circle_on_change(self,p,maxr=300,minr=30):
    '''
    ランダムに２つの日を抽出、さらにグラフの内側に円を描き、その内側の辺をランダムに交換する（maxr,minrは内側の円の大きさの最大値と最小値）。
    '''
    a=randint(0,self.D-1)
    b=randint(0,self.D-2)
    if b>=a: b+=1
    an,bn=self.cnt[a],self.cnt[b]
    r,theta,r2=uniform(0,1),uniform(0,1)*2*math.pi,minr+uniform(0,1)*(maxr-minr)
    x,y=500+500*r*math.cos(theta),500+500*r*math.sin(theta)
    tmpa,tmpb=[1 if self.ans[i]==a else 0 for i in range(self.M)],[1 if self.ans[i]==b else 0 for i in range(self.M)]
    modifya,modifyb=[],[]
    tmp=[(self.points[i][0]-x)**2+(self.points[i][1]-y)**2<r2**2 for i in range(self.N)]
    for i in range(self.M):
      w,u,v,k=self.E[i]
      if tmp[u]*tmp[v]==True and uniform(0,1)<p:
        if tmpa[k]==1: tmpa[k]=0; tmpb[k]=1; an-=1; bn+=1; modifya.append(k)
        elif tmpb[k]==1: tmpa[k]=1; tmpb[k]=0; an+=1; bn-=1; modifyb.append(k)
    if an>self.K:
      for i in range(an-self.K):
        x=modifyb.pop(); tmpa[x]=0; tmpb[x]=1; an-=1; bn+=1
    if bn>self.K:
      for i in range(bn-self.K):
        x=modifya.pop(); tmpa[x]=1; tmpb[x]=0; an+=1; bn-=1
    return modifya+modifyb,a,b,self.dijkstra(tmpa,self.djk_start),self.dijkstra(tmpb,self.djk_start)

  def anealing(self,start_temp,end_temp):
    cnt=1
    eps=1
    start_time=time.perf_counter()
    TIME_LIMIT=self.TL-start_time+self.start
    n=self.K-self.M//self.D
    # score_list=[]
    while True:
      now=time.perf_counter()- start_time
      if now > TIME_LIMIT: break
      #ダイクストラの始点追加（スコア関数の精緻化）
      if len(self.djk_start)==1 and now>2: self.add_start_point(self.djk_start_cand[1:2])
      elif len(self.djk_start)==2 and now>4: self.add_start_point(self.djk_start_cand[2:3])
      #random_change,random_change2,exchange,line_change,line_on_change,circle_change,circle_on_changeから組み合わせる
      p=uniform(0,1)
      if n>=10:
        if now<1: modify,a,b,score_a,score_b=self.line_on_change(0.2)
        elif now<3: modify,a,b,score_a,score_b=self.circle_change(0.05,200,100)
        else: modify,a,b,score_a,score_b=self.random_change(randint(4,6))
      else: modify,a,b,score_a,score_b=self.exchange(randint(4,6))
      temp = start_temp + (end_temp - start_temp) * now / TIME_LIMIT
      score=score_a+score_b-self.score[a]-self.score[b]
      # if score>0 and score<10**8: score_list.append(score)
      # 焼きなまし
      if score<0: prob=1
      elif (score+eps)/temp>10: prob=0
      else: prob = math.exp(-(score+eps)/temp)
      # 山登り
      # if score<0: prob=1
      # else: prob=0
      if prob>uniform(0,1):
        for i in range(len(modify)):
          if self.ans[modify[i]]==a: self.ans[modify[i]]=b; self.cnt[a]-=1; self.cnt[b]+=1
          elif self.ans[modify[i]]==b: self.ans[modify[i]]=a; self.cnt[a]+=1; self.cnt[b]-=1
        self.score[a],self.score[b]=score_a,score_b
        self.changed+=1
        self.updated=now
      cnt+=1
    # print(min(score_list),max(score_list),self.N,self.M)

  def solve(self):
    self.add_start_point([self.djk_start_cand[0]])
    self.init_ans2()
    for i in range(0): #ここはやってもダメっぽかった
      ans_tmp,cnt_tmp,score_tmp=self.init_ans_repeat()
      if sum(score_tmp)<sum(self.score):
        self.ans,self.cnt,self.score=ans_tmp,cnt_tmp,score_tmp
    self.anealing(20000,0)

def submit():
  start = time.perf_counter()
  TL = 5.7
  N,M,D,K=map(int,input().split())
  G=[[] for _ in range(N)]
  E=[]
  for i in range(M):
    u,v,w=map(int,input().split())
    u-=1; v-=1
    G[u].append((w,v,i))
    G[v].append((w,u,i))
    E.append((w,u,v,i))
  points=[list(map(int,input().split())) for _ in range(N)]
  tmp=[[500,500],[0,0],[1000,1000],[1000,0],[0,1000],[500,0],[500,1000],[0,500],[1000,500]]
  dist=[500000]*len(tmp)
  djk_start=[0]*len(tmp)
  for i in range(N):
    x,y=points[i]
    for j in range(len(tmp)):
      if (x-tmp[j][0])**2+(y-tmp[j][1])**2<dist[j]: djk_start[j]=i; dist[j]=(x-tmp[j][0])**2+(y-tmp[j][1])**2
  C=Ahc017(start,TL,N,M,D,K,G,E,points,djk_start)
  C.solve()
  print(*[C.ans[i]+1 for i in range(M)])
  # print(C.changed,C.updated)
  # print(time.perf_counter()-start)

def test(N):
  for i in range(N):
    N,M,D,K=map(int,input().split())
    for j in range(M):
      u,v,w=map(int,input().split())
    points=[list(map(int,input().split())) for _ in range(N)]
  submit()

if __name__ == "__main__":
  submit()