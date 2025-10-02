// Smooth scroll for nav
document.querySelectorAll('.nav a').forEach(a=>{
  a.addEventListener('click', function(e){
    e.preventDefault();
    const id = this.getAttribute('href');
    document.querySelector(id).scrollIntoView({behavior:'smooth', block:'start'});
  });
});

// Simple placeholder: show alert on contact submit (no backend)
const form = document.querySelector('.contact-form');
if(form) form.addEventListener('submit', function(e){
  e.preventDefault();
  alert('Terima kasih! Pesan Anda telah terkirim (simulasi).');
  form.reset();
});
