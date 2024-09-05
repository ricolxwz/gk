function umami() {
  if (window.location.hostname !== "localhost" && window.location.hostname !== "127.0.0.1") {
    var script = document.createElement("script");
    script.defer = true;
    script.src = "https://umami.ricolxwz.io/script.js";
    script.setAttribute(
      "data-website-id",
      "e70268ae-b4d1-49e4-b8ef-8fa95a12a8f2"
    );
    document.head.appendChild(script);
  }
}
umami();