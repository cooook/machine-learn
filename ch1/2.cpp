# include <stdio.h>  
int State[50], cnt, Sum[20];
bool vis[1 << 18];


void dfs(int k, int start, int now, int S) {
    if (now == k) {
        if (vis[S]) return ;
        vis[S] = true;
        ++Sum[k];
        return ;
    }
    for (int i = start; i < cnt; ++i)
        dfs(k, i + 1, now + 1, S | State[i]); 
}

int main(){
    int S1, S2, S3, i, j, k, i1, j1, k1; 
    for (i = 0; i <= 3; ++i)
        for (j = 0; j <= 3; ++j)
            for (k = 0; k <= 2; ++k) {
                S1 = (!i) ? 7 : 1 << (i - 1);
                S2 = (!j) ? 7 : 1 << (j - 1);
                S3 = (!k) ? 3 : 1 << (k - 1);
                for (i1 = 0; i1 <= 2; ++i1)
                    if (S1 >> i1 & 1)
                        for (j1 = 0; j1 <=2; ++j1)
                            if (S2 >> j1 & 1)
                                for (k1 = 0; k1 <= 1; ++k1)
                                    if (S3 >> k1 & 1)
                                        State[cnt] |= 1 << (i1 * 6 + j1 * 2 + k1);
                ++ cnt; 
            }
    Sum[0] = 1; 
    for (i = 1; i <= 18; ++i) {
        dfs(i, 0, 0, 0);
        Sum[i] += Sum[i - 1];
        printf("%d %d\n", i, Sum[i]);
        if (Sum[i] == 1 << 18)
            break; 
    }
    ++i;
    for (; i <= 18; ++i)
        Sum[i] += Sum[i - 1];
    for (i = 0; i <= 18; ++i)
        printf("%d %d\n", i, Sum[i]);
    return 0; 
}
