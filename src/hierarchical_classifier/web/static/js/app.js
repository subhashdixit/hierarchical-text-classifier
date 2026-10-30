"use strict";
const examples = {
  dog: [
    "A tough chew toy",
    "This rubber dog toy survives chewing and keeps my puppy entertained.",
  ],
  vitamin: [
    "Daily vitamin supplement",
    "These vitamin tablets support my daily nutrition and wellness routine.",
  ],
  beauty: [
    "Gentle face moisturizer",
    "This skin care cream leaves my face soft and hydrated without a greasy feeling.",
  ],
};
document.querySelectorAll("[data-example]").forEach((button) => {
  button.addEventListener("click", () => {
    const [title, text] = examples[button.dataset.example];
    document.getElementById("title").value = title;
    document.getElementById("text").value = text;
    document.getElementById("text").focus();
  });
});
