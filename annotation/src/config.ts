// The deployed build receives the Apps Script web app URL as VITE_ENDPOINT (GitHub repository variable APPS_SCRIPT_URL).
// If empty, requests go to the local test server (/api).
export const ENDPOINT: string = import.meta.env.VITE_ENDPOINT || "/api";
export const DATA = `${import.meta.env.BASE_URL}data/`;
