/** Typed access to the fact knowledge layer API. */
async function get(path) {
    const response = await fetch(path);
    if (!response.ok) {
        throw new Error(`${response.status} ${response.statusText}`);
    }
    return (await response.json());
}
export const api = {
    documents: () => get("/api/documents"),
    stats: () => get("/api/stats"),
    clusters: () => get("/api/clusters"),
    job: (id) => get(`/api/jobs/${id}`),
    facts: (params = {}) => {
        const query = new URLSearchParams(params).toString();
        return get(`/api/facts?${query}`);
    },
    fact: (id) => get(`/api/facts/${id}`),
    conflicts: (verdict) => get(`/api/conflicts${verdict ? `?verdict=${verdict}` : ""}`),
    async upload(file) {
        const body = new FormData();
        body.append("file", file);
        const response = await fetch("/api/documents", { method: "POST", body });
        if (!response.ok) {
            throw new Error(await response.text());
        }
        return (await response.json());
    },
};
