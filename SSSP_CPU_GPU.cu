#include<bits/stdc++.h>
#include<cuda.h>

using namespace std;

#define INF INT_MAX

void cpu_SSSP(int V, vector<int>& row, vector<int>& col, vector<int>& wt, vector<int>& dist){
    dist = vector<int>(V, INF);
    dist[0] = 0;

    for(int i=0; i<V-1;i++){
        bool change = false;

        for(int u=0;u<V;u++){
          for(int j=row[u];j<row[u+1];j++){
            int v = col[j];

            if(dist[u] != INF && dist[u] + wt[j] < dist[v]){
                dist[v] = dist[u] + wt[j];
                change = true;
            }
          }
        }
        if(!change){
            break;
        }
    }
}

__global__ void gpu_SSSP(int V, int *row, int *col, int *wt, int *dist, int *newDist){
    int u = blockIdx.x * blockDim.x + threadIdx.x;

    if(u<V && dist[u] != INF){
        for(int j = row[u];j<row[u+1];j++){
          int v = col[j];
          int d = dist[u] + wt[j];

          if(d < newDist[v]){
              atomicMin(&newDist[v], d);
          }
        }
    }
}

int main() {
    int V = 5, E = 6;
    int source = 0;

    int src[] = {0, 0, 1, 1, 2, 3};
    int dest[] = {1, 2, 3, 4, 4, 4};
    int weight[] = {2, 4, 3, 1, 2, 5};

    vector<int> row(V + 1, 0);

    for (int i = 0; i < E; i++) {
        row[src[i] + 1]++;
    }

    for (int i = 1; i <= V; i++) {
        row[i] += row[i - 1];
    }

    vector<int> col(E), wt(E);
    vector<int> pos = row;

    for (int i = 0; i < E; i++) {
        int p = pos[src[i]]++;
        col[p] = dest[i];
        wt[p] = weight[i];
    }

    vector<int> cpuDist;
    cpu_SSSP(V, row, col, wt, cpuDist);

    int *d_row, *d_col, *d_wt, *d_dist, *d_newDist;

    cudaMalloc(&d_row, (V + 1) * sizeof(int));
    cudaMalloc(&d_col, E * sizeof(int));
    cudaMalloc(&d_wt, E * sizeof(int));
    cudaMalloc(&d_dist, V * sizeof(int));
    cudaMalloc(&d_newDist, V * sizeof(int));

    cudaMemcpy(d_row, row.data(), (V + 1) * sizeof(int),
               cudaMemcpyHostToDevice);
    cudaMemcpy(d_col, col.data(), E * sizeof(int),
               cudaMemcpyHostToDevice);
    cudaMemcpy(d_wt, wt.data(), E * sizeof(int),
               cudaMemcpyHostToDevice);

    vector<int> initialDist(V, INF);
    initialDist[source] = 0;

    cudaMemcpy(d_dist, initialDist.data(), V * sizeof(int),cudaMemcpyHostToDevice);


    for (int i = 0; i < V - 1; i++) {
        cudaMemcpy(d_newDist, d_dist, V * sizeof(int),cudaMemcpyDeviceToDevice);

        gpu_SSSP<<<(V + 255) / 256, 256>>>(V, d_row, d_col, d_wt, d_dist, d_newDist);

        swap(d_dist, d_newDist);
    }

    cudaDeviceSynchronize();


    vector<int> gpuDist(V, INF);
    cudaMemcpy(gpuDist.data(), d_dist, V * sizeof(int),cudaMemcpyDeviceToHost);

    cout << "Vertex\tCPU\tGPU\n";
    for (int i = 0; i < V; i++) {
        cout << i << "\t" << cpuDist[i]
             << "\t" << gpuDist[i] << endl;
    }

    bool same = true;
    for (int i = 0; i < V; i++) {
        if (cpuDist[i] != gpuDist[i]) {
            same = false;
            break;
        }
    }

    if (same)
        cout << "Results are same" << endl;
    else
        cout << "Results are different" << endl;

    cudaFree(d_row);
    cudaFree(d_col);
    cudaFree(d_wt);
    cudaFree(d_dist);
    cudaFree(d_newDist);

    return 0;
}

