(()=>{var Ir=Object.create;var Fe=Object.defineProperty;var Ar=Object.getOwnPropertyDescriptor;var Sr=Object.getOwnPropertyNames;var Br=Object.getPrototypeOf,Mr=Object.prototype.hasOwnProperty;var Lt=n=>e=>{var s=n[e];if(s)return s();throw new Error("Module not found in bundle: "+e)};var ee=(n,e)=>()=>(n&&(e=n(n=0)),e);var _t=(n,e)=>()=>(e||n((e={exports:{}}).exports,e),e.exports),te=(n,e)=>{for(var s in e)Fe(n,s,{get:e[s],enumerable:!0})},qr=(n,e,s,o)=>{if(e&&typeof e=="object"||typeof e=="function")for(let h of Sr(e))!Mr.call(n,h)&&h!==s&&Fe(n,h,{get:()=>e[h],enumerable:!(o=Ar(e,h))||o.enumerable});return n};var Pr=(n,e,s)=>(s=n!=null?Ir(Br(n)):{},qr(e||!n||!n.__esModule?Fe(s,"default",{value:n,enumerable:!0}):s,n));function j(...n){return n.flat().filter(e=>e&&typeof e=="string").join(" ")}function O(n="div",{id:e="",className:s="",attrs:o={},html:h="",text:w=""}={}){let _=document.createElement(n);return e&&(_.id=e),s&&(_.className=s),w&&(_.textContent=w),h&&(_.innerHTML=h),Object.entries(o).forEach(([L,A])=>{A===!0?_.setAttribute(L,""):A!==!1&&A!==null&&_.setAttribute(L,A)}),_}function $t(n,e,s,o={}){if(n)return n.addEventListener(e,s,o),()=>n.removeEventListener(e,s,o)}function Pe(n){let e=n.querySelectorAll('button, [href], input, select, textarea, [tabindex]:not([tabindex="-1"])'),s=e[0],o=e[e.length-1];return{init(){n.addEventListener("keydown",h=>{h.key==="Tab"&&(h.shiftKey?document.activeElement===s&&(h.preventDefault(),o.focus()):document.activeElement===o&&(h.preventDefault(),s.focus()))})},destroy(){}}}function Tt(){return"id-"+Math.random().toString(36).substr(2,9)+Date.now().toString(36)}function Se(n=""){return O("div",{className:j("fixed inset-0 bg-black bg-opacity-50 z-40 transition-opacity duration-200",n),attrs:{"data-backdrop":"true"}})}var He,oe=ee(()=>{He={isEnter:n=>n.key==="Enter",isEscape:n=>n.key==="Escape",isArrowUp:n=>n.key==="ArrowUp",isArrowDown:n=>n.key==="ArrowDown",isArrowLeft:n=>n.key==="ArrowLeft",isArrowRight:n=>n.key==="ArrowRight",isSpace:n=>n.key===" ",isTab:n=>n.key==="Tab"}});var V,ie=ee(()=>{oe();V=class n{constructor(e={}){this.id=e.id||Tt(),this.element=null,this.listeners=[],this.isInitialized=!1,this.config=e}create(e="div",{className:s="",attrs:o={},html:h=""}={}){return this.element=O(e,{id:this.id,className:s,attrs:o,html:h}),this.element}mount(e){if(!this.element)return!1;let s=typeof e=="string"?document.querySelector(e):e;return s?(s.appendChild(this.element),this.isInitialized=!0,!0):!1}on(e,s,o={}){if(!this.element)return;let h=$t(this.element,e,s,o);return this.listeners.push(h),h}delegate(e,s,o){if(!this.element)return;let h=w=>{let _=w.target.closest(e);_&&o.call(_,w)};this.element.addEventListener(s,h),this.listeners.push(()=>this.element.removeEventListener(s,h))}addClass(...e){this.element&&this.element.classList.add(...e)}removeClass(...e){this.element&&this.element.classList.remove(...e)}toggleClass(e,s){this.element&&this.element.classList.toggle(e,s)}hasClass(e){return this.element?.classList.contains(e)??!1}attr(e,s){if(this.element){if(s===void 0)return this.element.getAttribute(e);s===null||s===!1?this.element.removeAttribute(e):s===!0?this.element.setAttribute(e,""):this.element.setAttribute(e,s)}}attrs(e){Object.entries(e).forEach(([s,o])=>{this.attr(s,o)})}text(e){this.element&&(this.element.textContent=e)}html(e){this.element&&(this.element.innerHTML=e)}append(e){this.element&&e&&this.element.appendChild(e instanceof n?e.element:e)}prepend(e){this.element&&e&&this.element.prepend(e instanceof n?e.element:e)}show(){this.element&&(this.element.style.display="",this.element.removeAttribute("hidden"))}hide(){this.element&&(this.element.style.display="none")}toggle(e){this.element&&(e===void 0&&(e=this.element.style.display==="none"),e?this.show():this.hide())}getStyle(e){return this.element?window.getComputedStyle(this.element).getPropertyValue(e):null}setStyle(e,s){this.element&&(this.element.style[e]=s)}setStyles(e){Object.entries(e).forEach(([s,o])=>{this.setStyle(s,o)})}focus(e){if(this.element)try{typeof e>"u"?this.element.focus({preventScroll:!0}):this.element.focus(e)}catch{try{this.element.focus()}catch{}}}blur(){this.element&&this.element.blur()}getPosition(){return this.element?this.element.getBoundingClientRect():null}destroy(){this.listeners.forEach(e=>e?.()),this.listeners=[],this.element?.parentNode&&this.element.parentNode.removeChild(this.element),this.element=null,this.isInitialized=!1}init(){this.element&&!this.isInitialized&&(this.isInitialized=!0)}render(){return this.element}}});var It={};te(It,{Alert:()=>ze});var ze,At=ee(()=>{ie();oe();ze=class extends V{constructor(e={}){super(e),this.title=e.title||"",this.message=e.message||"",this.type=e.type||"default",this.icon=e.icon||null,this.closeable=e.closeable||!1,this.className=e.className||""}create(){let e={default:{bg:"bg-blue-50",border:"border-blue-200",title:"text-blue-900",message:"text-blue-800",icon:"\u24D8"},success:{bg:"bg-green-50",border:"border-green-200",title:"text-green-900",message:"text-green-800",icon:"\u2713"},warning:{bg:"bg-yellow-50",border:"border-yellow-200",title:"text-yellow-900",message:"text-yellow-800",icon:"\u26A0"},error:{bg:"bg-red-50",border:"border-red-200",title:"text-red-900",message:"text-red-800",icon:"\u2715"},info:{bg:"bg-cyan-50",border:"border-cyan-200",title:"text-cyan-900",message:"text-cyan-800",icon:"\u2139"}},s=e[this.type]||e.default,h=super.create("div",{className:j("p-4 rounded-lg border-2",s.bg,s.border,this.className),attrs:{role:"alert"}}),w="";return(this.title||this.icon)&&(w+='<div class="flex items-center gap-3 mb-2">',this.icon,w+=`<span class="text-xl font-bold ${s.title}">${this.icon||s.icon}</span>`,this.title&&(w+=`<h4 class="font-semibold ${s.title}">${this.title}</h4>`),w+="</div>"),this.message&&(w+=`<p class="${s.message}">${this.message}</p>`),this.closeable&&(w+='<button class="absolute top-4 right-4 text-gray-400 hover:text-gray-600" aria-label="Close alert">\xD7</button>'),h.innerHTML=w,this.closeable&&h.querySelector("button")?.addEventListener("click",()=>{this.destroy()}),h}setMessage(e){if(this.message=e,this.element){let s=this.element.querySelector("p");s&&(s.textContent=e)}}}});var St={};te(St,{AlertDialog:()=>Re});var Re,Bt=ee(()=>{ie();oe();Re=class extends V{constructor(e={}){super(e),this.title=e.title||"",this.message=e.message||"",this.confirmText=e.confirmText||"Confirm",this.cancelText=e.cancelText||"Cancel",this.type=e.type||"warning",this.onConfirm=e.onConfirm||null,this.onCancel=e.onCancel||null,this.open=e.open||!1}create(){let e=super.create("div",{className:j("fixed inset-0 z-50 flex items-center justify-center",this.open?"":"hidden"),attrs:{role:"alertdialog","aria-modal":"true","aria-labelledby":`${this.id}-title`,"aria-describedby":`${this.id}-message`}}),s=Se();e.appendChild(s);let o=document.createElement("div");o.className="bg-white rounded-lg shadow-lg relative z-50 w-full max-w-md mx-4";let h=document.createElement("div");h.className="px-6 py-4 border-b border-gray-200";let w=document.createElement("h2");w.id=`${this.id}-title`,w.className="text-lg font-semibold text-gray-900",w.textContent=this.title,h.appendChild(w),o.appendChild(h);let _=document.createElement("div");_.id=`${this.id}-message`,_.className="px-6 py-4 text-gray-700",_.textContent=this.message,o.appendChild(_);let L=document.createElement("div");L.className="px-6 py-4 bg-gray-50 border-t border-gray-200 flex gap-3 justify-end rounded-b-lg";let A=document.createElement("button");A.className="px-4 py-2 border border-gray-300 rounded-md text-gray-700 hover:bg-gray-100 transition-colors duration-200",A.textContent=this.cancelText,A.addEventListener("click",()=>this.handleCancel()),L.appendChild(A);let Y=document.createElement("button"),Z=this.type==="danger"?"bg-red-600 text-white hover:bg-red-700":"bg-blue-600 text-white hover:bg-blue-700";return Y.className=j("px-4 py-2 rounded-md transition-colors duration-200",Z),Y.textContent=this.confirmText,Y.addEventListener("click",()=>this.handleConfirm()),L.appendChild(Y),o.appendChild(L),e.appendChild(o),this.focusTrap=Pe(o),e}open(){this.element||this.create(),this.open=!0,this.element.classList.remove("hidden"),this.focusTrap.init(),document.body.style.overflow="hidden"}close(){this.open=!1,this.element?.classList.add("hidden"),document.body.style.overflow=""}handleConfirm(){this.onConfirm&&this.onConfirm(),this.close()}handleCancel(){this.onCancel&&this.onCancel(),this.close()}}});var Mt={};te(Mt,{Avatar:()=>Oe});var Oe,qt=ee(()=>{ie();oe();Oe=class extends V{constructor(e={}){super(e),this.src=e.src||"",this.alt=e.alt||"",this.initials=e.initials||"",this.size=e.size||"md",this.className=e.className||"",this.fallbackBg=e.fallbackBg||"bg-blue-600"}create(){let e={xs:"w-6 h-6 text-xs",sm:"w-8 h-8 text-sm",md:"w-10 h-10 text-base",lg:"w-12 h-12 text-lg",xl:"w-16 h-16 text-xl"},s="rounded-full overflow-hidden flex items-center justify-center flex-shrink-0 font-semibold";return this.src?super.create("img",{className:j(s,e[this.size],this.className),attrs:{src:this.src,alt:this.alt,role:"img"}}):this.initials?super.create("div",{className:j(s,e[this.size],"text-white",this.fallbackBg,this.className),text:this.initials.toUpperCase()}):super.create("div",{className:j(s,e[this.size],"bg-gray-300",this.className),html:'<svg class="w-full h-full text-gray-500" fill="currentColor" viewBox="0 0 24 24"><path d="M12 12c2.21 0 4-1.79 4-4s-1.79-4-4-4-4 1.79-4 4 1.79 4 4 4zm0 2c-2.67 0-8 1.34-8 4v2h16v-2c0-2.66-5.33-4-8-4z"/></svg>'})}setSrc(e,s=""){this.src=e,this.alt=s,this.element&&this.element.tagName==="IMG"&&(this.element.src=e,this.element.alt=s)}setInitials(e,s=""){if(this.initials=e,s&&(this.fallbackBg=s),this.element&&(this.element.textContent=e.toUpperCase(),s&&this.element.className.includes("bg-"))){let o=this.element.className.match(/bg-\S+/)[0];this.element.classList.remove(o),this.element.classList.add(s)}}}});var Pt={};te(Pt,{Badge:()=>Ve});var Ve,Ht=ee(()=>{ie();oe();Ve=class extends V{constructor(e={}){super(e),this.label=e.label||"Badge",this.variant=e.variant||"default",this.size=e.size||"md",this.className=e.className||""}create(){let e="inline-flex items-center rounded font-semibold whitespace-nowrap",s={default:"bg-gray-100 text-gray-800",primary:"bg-blue-100 text-blue-800",success:"bg-green-100 text-green-800",warning:"bg-yellow-100 text-yellow-800",destructive:"bg-red-100 text-red-800",outline:"border border-gray-300 text-gray-700"},o={sm:"px-2 py-1 text-xs",md:"px-3 py-1 text-sm",lg:"px-4 py-2 text-base"},h=super.create("span",{className:j(e,s[this.variant],o[this.size],this.className)});return h.textContent=this.label,h}setLabel(e){this.label=e,this.element&&(this.element.textContent=e)}}});var jt={};te(jt,{Button:()=>De});var De,Nt=ee(()=>{ie();oe();De=class extends V{constructor(e={}){super(e),this.label=e.label||"Button",this.variant=e.variant||"default",this.size=e.size||"md",this.disabled=e.disabled||!1,this.onClick=e.onClick||null,this.className=e.className||""}create(){let e="px-4 py-2 font-medium rounded transition-colors duration-200 disabled:opacity-50 disabled:cursor-not-allowed focus:outline-none focus:ring-2 focus:ring-offset-2",s={default:"bg-gray-200 text-gray-900 hover:bg-gray-300",primary:"bg-blue-600 text-white hover:bg-blue-700 focus:ring-blue-500",secondary:"bg-green-600 text-white hover:bg-green-700 focus:ring-green-500",destructive:"bg-red-600 text-white hover:bg-red-700 focus:ring-red-500",outline:"border-2 border-gray-300 text-gray-900 hover:bg-gray-50",ghost:"text-gray-900 hover:bg-gray-100"},o={sm:"px-2 py-1 text-sm",md:"px-4 py-2 text-base",lg:"px-6 py-3 text-lg"},h=super.create("button",{className:j(e,s[this.variant],o[this.size],this.className),attrs:{disabled:this.disabled}});return h.textContent=this.label,this.onClick&&this.on("click",this.onClick),h}setLabel(e){this.label=e,this.element&&(this.element.textContent=e)}setDisabled(e){this.disabled=e,this.attr("disabled",e?"":null)}setLoading(e){this.setDisabled(e),e?this.html('<span class="flex items-center gap-2"><span class="inline-block w-4 h-4 border-2 border-current border-t-transparent rounded-full animate-spin"></span>Loading...</span>'):this.text(this.label)}}});var Ft={};te(Ft,{ButtonGroup:()=>Ue});var Ue,zt=ee(()=>{ie();oe();Ue=class extends V{constructor(e={}){super(e),this.buttons=e.buttons||[],this.orientation=e.orientation||"horizontal",this.size=e.size||"md",this.className=e.className||""}create(){let e=this.orientation==="vertical"?"flex-col":"flex-row",o=super.create("div",{className:j("flex",e,"inline-flex border border-gray-300 rounded-md overflow-hidden",this.className),attrs:{role:"group"}});return this.buttons.forEach((h,w)=>{let _=document.createElement("button");_.textContent=h.label||"Button",_.className=j("px-4 py-2 font-medium text-gray-700 hover:bg-gray-50 transition-colors duration-200",w>0?this.orientation==="vertical"?"border-t border-gray-300":"border-l border-gray-300":"",h.disabled?"opacity-50 cursor-not-allowed":""),_.disabled=h.disabled||!1,h.onClick&&_.addEventListener("click",h.onClick),o.appendChild(_)}),o}addButton(e,s){if(this.element){let o=document.createElement("button");o.textContent=e,o.className=j("px-4 py-2 font-medium text-gray-700 hover:bg-gray-50 transition-colors duration-200 border-l border-gray-300"),s&&o.addEventListener("click",s),this.element.appendChild(o)}}}});var Rt={};te(Rt,{Breadcrumb:()=>We});var We,Ot=ee(()=>{ie();oe();We=class extends V{constructor(e={}){super(e),this.items=e.items||[],this.className=e.className||""}create(){let s=super.create("nav",{className:j("flex items-center gap-2",this.className),attrs:{"aria-label":"Breadcrumb"}});return this.items.forEach((o,h)=>{if(h>0){let w=O("span",{className:"text-gray-400 mx-1",text:"/"});s.appendChild(w)}if(h===this.items.length-1){let w=O("span",{className:"text-gray-700 font-medium",text:o.label,attrs:{"aria-current":"page"}});s.appendChild(w)}else{let w=O("a",{className:"text-blue-600 hover:text-blue-800 hover:underline transition-colors duration-200",text:o.label,attrs:{href:o.href||"#"}});s.appendChild(w)}}),s}addItem(e,s="#"){if(this.element){if(this.element.children.length>0){let h=O("span",{className:"text-gray-400 mx-1",text:"/"});this.element.appendChild(h)}let o=O("a",{className:"text-blue-600 hover:text-blue-800 hover:underline transition-colors duration-200",text:e,attrs:{href:s}});this.element.appendChild(o),this.items.push({label:e,href:s})}}}});var Vt={};te(Vt,{Card:()=>Ye});var Ye,Dt=ee(()=>{ie();oe();Ye=class extends V{constructor(e={}){super(e),this.title=e.title||"",this.subtitle=e.subtitle||"",this.content=e.content||"",this.footer=e.footer||"",this.className=e.className||"",this.hoverable=e.hoverable!==!1}create(){let e="bg-white rounded-lg border border-gray-200 overflow-hidden",s=this.hoverable?"hover:shadow-lg transition-shadow duration-300":"",o=super.create("div",{className:j(e,s,this.className)});if(this.title){let h=O("div",{className:"px-6 py-4 border-b border-gray-200 bg-gray-50"}),w=`<h3 class="text-lg font-semibold text-gray-900">${this.title}</h3>`;this.subtitle&&(w+=`<p class="text-sm text-gray-600 mt-1">${this.subtitle}</p>`),h.innerHTML=w,o.appendChild(h)}if(this.content){let h=O("div",{className:"px-6 py-4",html:this.content});o.appendChild(h)}if(this.footer){let h=O("div",{className:"px-6 py-4 border-t border-gray-200 bg-gray-50",html:this.footer});o.appendChild(h)}return o}setContent(e){if(this.content=e,this.element){let s=this.element.querySelector(".px-6.py-4:not(.border-b):not(.border-t)");s&&(s.innerHTML=e)}}addContent(e){if(this.element){let s=this.element.querySelector(".px-6.py-4:not(.border-b):not(.border-t)");s&&s.appendChild(e instanceof V?e.element:e)}}}});var Ut={};te(Ut,{Carousel:()=>Ge});var Ge,Wt=ee(()=>{ie();oe();Ge=class extends V{constructor(e={}){super(e),this.items=e.items||[],this.autoplay=e.autoplay||!0,this.interval=e.interval||5e3,this.className=e.className||"",this.currentIndex=0,this.autoplayTimer=null}create(){let e=super.create("div",{className:j("relative w-full bg-black rounded-lg overflow-hidden",this.className)}),s=O("div",{className:"relative w-full h-96 overflow-hidden"});this.items.forEach((_,L)=>{let A=O("div",{className:j("absolute inset-0 transition-opacity duration-500",L===this.currentIndex?"opacity-100":"opacity-0")}),Y=document.createElement("img");if(Y.src=_.src,Y.alt=_.alt||"",Y.className="w-full h-full object-cover",A.appendChild(Y),_.title){let Z=O("div",{className:"absolute bottom-0 left-0 right-0 bg-gradient-to-t from-black to-transparent p-4",html:`<p class="text-white font-semibold">${_.title}</p>`});A.appendChild(Z)}s.appendChild(A)}),e.appendChild(s);let o=O("button",{className:"absolute left-4 top-1/2 transform -translate-y-1/2 z-10 bg-black bg-opacity-50 text-white p-2 rounded-full hover:bg-opacity-75 transition-all",html:"\u276E"});o.addEventListener("click",()=>{this.previous()});let h=O("button",{className:"absolute right-4 top-1/2 transform -translate-y-1/2 z-10 bg-black bg-opacity-50 text-white p-2 rounded-full hover:bg-opacity-75 transition-all",html:"\u276F"});h.addEventListener("click",()=>{this.next()}),e.appendChild(o),e.appendChild(h);let w=O("div",{className:"absolute bottom-4 left-1/2 transform -translate-x-1/2 flex gap-2 z-10"});return this.items.forEach((_,L)=>{let A=O("button",{className:j("w-2 h-2 rounded-full transition-all",L===this.currentIndex?"bg-white w-8":"bg-gray-500"),attrs:{"data-index":L}});A.addEventListener("click",()=>{this.goTo(L)}),w.appendChild(A)}),e.appendChild(w),this.slidesElement=s,this.dotsElement=w,this.autoplay&&this.startAutoplay(),e}next(){this.currentIndex=(this.currentIndex+1)%this.items.length,this.updateSlides()}previous(){this.currentIndex=(this.currentIndex-1+this.items.length)%this.items.length,this.updateSlides()}goTo(e){this.currentIndex=e,this.updateSlides(),this.autoplay&&this.resetAutoplay()}updateSlides(){let e=this.slidesElement.querySelectorAll("div"),s=this.dotsElement.querySelectorAll("button");e.forEach((o,h)=>{o.className=j("absolute inset-0 transition-opacity duration-500",h===this.currentIndex?"opacity-100":"opacity-0")}),s.forEach((o,h)=>{o.className=j("w-2 h-2 rounded-full transition-all",h===this.currentIndex?"bg-white w-8":"bg-gray-500")})}startAutoplay(){this.autoplayTimer=setInterval(()=>{this.next()},this.interval)}stopAutoplay(){this.autoplayTimer&&clearInterval(this.autoplayTimer)}resetAutoplay(){this.stopAutoplay(),this.autoplay&&this.startAutoplay()}destroy(){this.stopAutoplay(),super.destroy()}}});var Yt={};te(Yt,{Chart:()=>Qe});var Qe,Gt=ee(()=>{ie();oe();Qe=class extends V{constructor(e={}){super(e),this.type=e.type||"bar",this.data=e.data||[],this.title=e.title||"",this.height=e.height||"300px",this.className=e.className||""}create(){let e=super.create("div",{className:j("bg-white rounded-lg border border-gray-200 p-4",this.className)});if(this.title){let o=O("h3",{className:"text-lg font-semibold text-gray-900 mb-4",text:this.title});e.appendChild(o)}let s=this.createSVG();return e.appendChild(s),e}createSVG(){let e=Math.max(...this.data.map(_=>_.value)),s=400,o=200,h=40,w=document.createElementNS("http://www.w3.org/2000/svg","svg");if(w.setAttribute("viewBox",`0 0 ${s} ${o}`),w.setAttribute("class","w-full"),this.type==="bar"){let _=(s-h*2)/this.data.length;this.data.forEach((L,A)=>{let Y=L.value/e*(o-h*2),Z=h+A*_+_*.25,z=o-h-Y,E=document.createElementNS("http://www.w3.org/2000/svg","rect");E.setAttribute("x",Z),E.setAttribute("y",z),E.setAttribute("width",_*.5),E.setAttribute("height",Y),E.setAttribute("fill","#3b82f6"),E.setAttribute("class","hover:fill-blue-700 transition-colors"),w.appendChild(E);let H=document.createElementNS("http://www.w3.org/2000/svg","text");H.setAttribute("x",Z+_*.25),H.setAttribute("y",o-10),H.setAttribute("text-anchor","middle"),H.setAttribute("font-size","12"),H.setAttribute("fill","#6b7280"),H.textContent=L.label,w.appendChild(H)})}return w}setData(e){if(this.data=e,this.element&&this.element.parentNode){let s=this.create();this.element.parentNode.replaceChild(s,this.element),this.element=s}}}});var Qt={};te(Qt,{Checkbox:()=>Je});var Je,Jt=ee(()=>{ie();oe();Je=class extends V{constructor(e={}){super(e),this.label=e.label||"",this.checked=e.checked||!1,this.disabled=e.disabled||!1,this.required=e.required||!1,this.name=e.name||"",this.className=e.className||"",this.onChange=e.onChange||null}create(){let s=super.create("div",{className:j("flex items-center gap-2",this.className)}),o="w-5 h-5 border-2 border-gray-300 rounded cursor-pointer transition-colors duration-200 checked:bg-blue-600 checked:border-blue-600 disabled:opacity-50 disabled:cursor-not-allowed",h=document.createElement("input");if(h.type="checkbox",h.className=o,h.checked=this.checked,h.disabled=this.disabled,h.required=this.required,this.name&&(h.name=this.name),s.appendChild(h),this.label){let w=document.createElement("label");w.className="cursor-pointer select-none",w.textContent=this.label,s.appendChild(w),w.addEventListener("click",()=>{h.click()})}return this.onChange&&h.addEventListener("change",this.onChange),this.inputElement=h,s}isChecked(){return this.inputElement?.checked||!1}setChecked(e){this.checked=e,this.inputElement&&(this.inputElement.checked=e)}setDisabled(e){this.disabled=e,this.inputElement&&(this.inputElement.disabled=e)}toggle(){this.setChecked(!this.isChecked())}}});var Xt={};te(Xt,{Collapsible:()=>Xe});var Xe,Kt=ee(()=>{ie();oe();Xe=class extends V{constructor(e={}){super(e),this.title=e.title||"",this.content=e.content||"",this.open=e.open||!1,this.className=e.className||"",this.onChange=e.onChange||null}create(){let e=super.create("div",{className:j("border border-gray-200 rounded-lg overflow-hidden",this.className)}),s=O("button",{className:j("w-full px-4 py-3 flex items-center justify-between","hover:bg-gray-50 transition-colors duration-200 text-left"),attrs:{"aria-expanded":this.open}}),o=O("span",{className:"font-semibold text-gray-900",text:this.title}),h=O("span",{className:j("w-5 h-5 transition-transform duration-300",this.open?"rotate-180":""),html:"\u25BC"});s.appendChild(o),s.appendChild(h),s.addEventListener("click",()=>{this.toggle()}),e.appendChild(s);let w=O("div",{className:j("overflow-hidden transition-all duration-300",this.open?"max-h-96":"max-h-0","border-t border-gray-200")}),_=O("div",{className:"px-4 py-3",html:this.content});return w.appendChild(_),e.appendChild(w),this.triggerElement=s,this.contentElement=w,this.chevron=h,e}toggle(){this.open=!this.open,this.updateUI(),this.onChange&&this.onChange(this.open)}open(){this.open||(this.open=!0,this.updateUI(),this.onChange&&this.onChange(!0))}close(){this.open&&(this.open=!1,this.updateUI(),this.onChange&&this.onChange(!1))}updateUI(){this.triggerElement.setAttribute("aria-expanded",this.open),this.contentElement.className=j("overflow-hidden transition-all duration-300",this.open?"max-h-96":"max-h-0","border-t border-gray-200"),this.chevron.className=j("w-5 h-5 transition-transform duration-300",this.open?"rotate-180":"")}setContent(e){if(this.content=e,this.contentElement){let s=this.contentElement.querySelector("div");s&&(s.innerHTML=e)}}}});var Zt={};te(Zt,{Command:()=>Ke});var Ke,es=ee(()=>{ie();oe();Ke=class extends V{constructor(e={}){super(e),this.commands=e.commands||[],this.placeholder=e.placeholder||"Type a command...",this.className=e.className||"",this.open=!1}create(){let e=super.create("div",{className:j("fixed inset-0 z-50 hidden flex items-start justify-center pt-20",this.open?"flex":"")}),s=O("div",{className:"absolute inset-0 bg-black bg-opacity-50"});e.appendChild(s),s.addEventListener("click",()=>this.close());let o=O("div",{className:"relative w-full max-w-md bg-white rounded-lg shadow-lg z-50"}),h=document.createElement("input");h.type="text",h.placeholder=this.placeholder,h.className="w-full px-4 py-3 border-b border-gray-200 focus:outline-none",h.autofocus=!0,o.appendChild(h);let w=O("div",{className:"max-h-96 overflow-y-auto"}),_=(L="")=>{w.innerHTML="";let A=L?this.commands.filter(Z=>Z.label.toLowerCase().includes(L.toLowerCase())):this.commands;if(A.length===0){let Z=O("div",{className:"px-4 py-3 text-sm text-gray-500",text:"No commands found"});w.appendChild(Z);return}let Y="";A.forEach(Z=>{if(Z.category&&Z.category!==Y){Y=Z.category;let H=O("div",{className:"px-4 py-2 text-xs font-semibold text-gray-600 bg-gray-50 uppercase",text:Y});w.appendChild(H)}let z=O("div",{className:j("px-4 py-2 cursor-pointer hover:bg-blue-50 transition-colors flex items-center justify-between group")}),E=O("span",{text:Z.label,className:"text-sm text-gray-900"});if(z.appendChild(E),Z.shortcut){let H=O("span",{className:"text-xs text-gray-500 group-hover:text-gray-700",text:Z.shortcut});z.appendChild(H)}z.addEventListener("click",()=>{Z.action&&Z.action(),this.close()}),w.appendChild(z)})};return h.addEventListener("input",L=>{_(L.target.value)}),o.appendChild(w),e.appendChild(o),h.addEventListener("keydown",L=>{L.key==="Escape"&&this.close()}),_(),this.containerElement=e,e}open(){this.element||this.create(),this.open=!0,this.element.classList.remove("hidden"),this.element.classList.add("flex")}close(){this.open=!1,this.element?.classList.remove("flex"),this.element?.classList.add("hidden")}toggle(){this.open?this.close():this.open()}}});var ts={};te(ts,{Combobox:()=>Ze});var Ze,ss=ee(()=>{ie();oe();Ze=class extends V{constructor(e={}){super(e),this.items=e.items||[],this.value=e.value||"",this.placeholder=e.placeholder||"Search...",this.className=e.className||"",this.onChange=e.onChange||null,this.open=!1}create(){let e=super.create("div",{className:"relative w-full"}),s=document.createElement("input");s.type="text",s.placeholder=this.placeholder,s.value=this.value,s.className=j("w-full px-3 py-2 border border-gray-300 rounded-md","focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent");let o=O("div",{className:j("absolute hidden top-full left-0 right-0 mt-1 bg-white border border-gray-300 rounded-md shadow-lg z-50","max-h-64 overflow-y-auto")}),h=(w="")=>{o.innerHTML="";let _=this.items.filter(L=>L.label.toLowerCase().includes(w.toLowerCase()));if(_.length===0){let L=O("div",{className:"px-3 py-2 text-gray-500",text:"No results found"});o.appendChild(L);return}_.forEach(L=>{let A=O("div",{className:j("px-3 py-2 cursor-pointer hover:bg-blue-50 transition-colors",L.value===this.value?"bg-blue-100":""),text:L.label,attrs:{"data-value":L.value}});A.addEventListener("click",()=>{this.value=L.value,s.value=L.label,o.classList.add("hidden"),this.onChange&&this.onChange(this.value,L)}),o.appendChild(A)})};return s.addEventListener("input",w=>{h(w.target.value),o.classList.remove("hidden")}),s.addEventListener("focus",()=>{h(s.value),o.classList.remove("hidden")}),s.addEventListener("blur",()=>{setTimeout(()=>{o.classList.add("hidden")},150)}),e.appendChild(s),e.appendChild(o),document.addEventListener("click",w=>{e.contains(w.target)||o.classList.add("hidden")}),this.inputElement=s,this.listElement=o,h(),e}getValue(){return this.value}setValue(e){this.value=e;let s=this.items.find(o=>o.value===e);s&&this.inputElement&&(this.inputElement.value=s.label)}}});var rs={};te(rs,{ContextMenu:()=>et});var et,as=ee(()=>{ie();oe();et=class extends V{constructor(e={}){super(e),this.items=e.items||[],this.className=e.className||"",this.visible=!1}create(){let e=super.create("div",{className:"relative inline-block w-full"}),s=document.createElement("div");return s.className=j("absolute hidden bg-white border border-gray-200 rounded-lg shadow-lg py-1 z-50 min-w-max",this.className),this.items.forEach(o=>{let h=O("button",{className:j("w-full text-left px-4 py-2 hover:bg-gray-100 transition-colors duration-200 flex items-center gap-2",o.disabled?"opacity-50 cursor-not-allowed":"cursor-pointer"),attrs:{disabled:o.disabled?"":null,"data-action":o.label}});if(o.icon){let _=document.createElement("span");_.innerHTML=o.icon,h.appendChild(_)}let w=document.createElement("span");w.textContent=o.label,h.appendChild(w),h.addEventListener("click",()=>{!o.disabled&&o.onClick&&o.onClick(),this.hide()}),s.appendChild(h)}),e.appendChild(s),e.addEventListener("contextmenu",o=>{o.preventDefault(),this.showAt(o.clientX,o.clientY,s)}),document.addEventListener("click",()=>{this.visible&&this.hide()}),this.menuElement=s,e}showAt(e,s,o){o&&(this.visible=!0,o.classList.remove("hidden"),o.style.position="fixed",o.style.left=e+"px",o.style.top=s+"px")}hide(){this.visible=!1,this.menuElement&&(this.menuElement.classList.add("hidden"),this.menuElement.style.position="absolute")}addItem(e,s,o=null){let h={label:e,onClick:s,icon:o};if(this.items.push(h),this.menuElement){let w=O("button",{className:"w-full text-left px-4 py-2 hover:bg-gray-100 transition-colors duration-200 flex items-center gap-2"});if(o){let _=document.createElement("span");_.innerHTML=o,w.appendChild(_)}w.textContent=e,w.addEventListener("click",()=>{s(),this.hide()}),this.menuElement.appendChild(w)}}}});var ns={};te(ns,{DatePicker:()=>tt});var tt,os=ee(()=>{ie();oe();tt=class extends V{constructor(e={}){super(e),this.value=e.value||"",this.placeholder=e.placeholder||"Select date...",this.format=e.format||"yyyy-mm-dd",this.className=e.className||"",this.onChange=e.onChange||null,this.open=!1}create(){let e=super.create("div",{className:"relative w-full"}),s=document.createElement("input");s.type="text",s.placeholder=this.placeholder,s.value=this.value,s.className=j("w-full px-3 py-2 border border-gray-300 rounded-md","focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent"),s.addEventListener("click",()=>{this.openPicker()}),s.addEventListener("change",h=>{this.value=h.target.value,this.onChange&&this.onChange(this.value)}),e.appendChild(s);let o=document.createElement("input");return o.type="date",o.style.display="none",o.value=this.value,o.addEventListener("change",h=>{let w=new Date(h.target.value);this.value=this.formatDate(w),s.value=this.value,this.onChange&&this.onChange(this.value)}),e.appendChild(o),this.inputElement=s,this.nativeInput=o,e}openPicker(){this.nativeInput.click()}formatDate(e){let s=e.getFullYear(),o=String(e.getMonth()+1).padStart(2,"0"),h=String(e.getDate()).padStart(2,"0");return this.format==="dd/mm/yyyy"?`${h}/${o}/${s}`:`${s}-${o}-${h}`}getValue(){return this.value}setValue(e){this.value=e,this.inputElement&&(this.inputElement.value=e)}}});var is={};te(is,{Dialog:()=>st});var st,ls=ee(()=>{ie();oe();st=class extends V{constructor(e={}){super(e),this.title=e.title||"",this.content=e.content||"",this.size=e.size||"md",this.open=e.open||!1,this.onClose=e.onClose||null,this.closeButton=e.closeButton!==!1,this.closeOnBackdrop=e.closeOnBackdrop!==!1,this.closeOnEscape=e.closeOnEscape!==!1}create(){let e=super.create("div",{className:j("fixed inset-0 z-50 flex items-center justify-center",this.open?"":"hidden"),attrs:{role:"dialog","aria-modal":"true","aria-labelledby":`${this.id}-title`,"aria-describedby":`${this.id}-description`}}),s=Se("dialog-backdrop");e.appendChild(s),this.closeOnBackdrop&&s.addEventListener("click",()=>this.close());let o={sm:"w-full max-w-sm",md:"w-full max-w-md",lg:"w-full max-w-lg",xl:"w-full max-w-xl"},h=document.createElement("div");if(h.className=j("bg-white rounded-lg shadow-lg relative z-50",o[this.size],"mx-4 max-h-[90vh] overflow-y-auto"),this.title){let w=document.createElement("div");w.className="px-6 py-4 border-b border-gray-200 flex items-center justify-between";let _=document.createElement("h2");if(_.id=`${this.id}-title`,_.className="text-xl font-semibold text-gray-900",_.textContent=this.title,w.appendChild(_),this.closeButton){let L=document.createElement("button");L.className="text-gray-500 hover:text-gray-700 transition-colors duration-200",L.innerHTML="\xD7",L.setAttribute("aria-label","Close"),L.addEventListener("click",()=>this.close()),w.appendChild(L)}h.appendChild(w)}if(this.content){let w=document.createElement("div");w.id=`${this.id}-description`,w.className="px-6 py-4",w.innerHTML=this.content,h.appendChild(w)}return e.appendChild(h),this.on("keydown",w=>{He.isEscape(w)&&this.closeOnEscape&&this.close()},{once:!1}),this.backdrop=s,this.dialogElement=h,this.focusTrap=Pe(h),e}open(){this.element||this.create(),this.open=!0,this.element.classList.remove("hidden"),this.focusTrap.init(),document.body.style.overflow="hidden"}close(){this.open=!1,this.element?.classList.add("hidden"),document.body.style.overflow="",this.onClose&&this.onClose()}toggle(){this.open?this.close():this.open()}setContent(e){if(this.content=e,this.dialogElement){let s=this.dialogElement.querySelector(`#${this.id}-description`);s&&(s.innerHTML=e)}}}});var ds={};te(ds,{DropdownMenu:()=>rt});var rt,cs=ee(()=>{ie();oe();rt=class extends V{constructor(e={}){super(e),this.trigger=e.trigger||"Menu",this.items=e.items||[],this.position=e.position||"bottom",this.align=e.align||"left",this.className=e.className||"",this.open=!1}create(){let e=super.create("div",{className:"relative inline-block"}),s=document.createElement("button");s.className="px-4 py-2 bg-gray-100 text-gray-900 rounded hover:bg-gray-200 transition-colors duration-200 flex items-center gap-2",s.innerHTML=`${this.trigger} <svg class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M19 14l-7 7m0 0l-7-7m7 7V3"></path></svg>`,s.addEventListener("click",()=>this.toggle()),e.appendChild(s);let o=this.position==="top"?"bottom-full mb-2":"top-full mt-2",h=this.align==="right"?"right-0":"left-0",w=document.createElement("div");return w.className=j("absolute hidden bg-white border border-gray-200 rounded-lg shadow-lg py-1 z-50 min-w-max",o,h,this.className),this.items.forEach(_=>{if(_.divider){let Y=document.createElement("div");Y.className="border-t border-gray-200 my-1",w.appendChild(Y);return}let L=O("button",{className:j("w-full text-left px-4 py-2 hover:bg-gray-100 transition-colors duration-200 flex items-center gap-2",_.disabled?"opacity-50 cursor-not-allowed":"cursor-pointer"),attrs:{disabled:_.disabled?"":null}});if(_.icon){let Y=document.createElement("span");Y.innerHTML=_.icon,Y.className="w-4 h-4",L.appendChild(Y)}let A=document.createElement("span");A.textContent=_.label,L.appendChild(A),L.addEventListener("click",()=>{!_.disabled&&_.onClick&&_.onClick(),this.close()}),w.appendChild(L)}),e.appendChild(w),document.addEventListener("click",_=>{!e.contains(_.target)&&this.open&&this.close()}),this.triggerBtn=s,this.menuElement=w,e}toggle(){this.open?this.close():this.open()}open(){this.open=!0,this.menuElement.classList.remove("hidden")}close(){this.open=!1,this.menuElement.classList.add("hidden")}addItem(e,s,o=null){let h={label:e,onClick:s,icon:o};if(this.items.push(h),this.menuElement){let w=O("button",{className:"w-full text-left px-4 py-2 hover:bg-gray-100 transition-colors duration-200 flex items-center gap-2"});if(o){let L=document.createElement("span");L.innerHTML=o,L.className="w-4 h-4",w.appendChild(L)}let _=document.createElement("span");_.textContent=e,w.appendChild(_),w.addEventListener("click",()=>{s(),this.close()}),this.menuElement.appendChild(w)}}}});var us={};te(us,{Drawer:()=>at});var at,ms=ee(()=>{ie();oe();at=class extends V{constructor(e={}){super(e),this.title=e.title||"",this.content=e.content||"",this.position=e.position||"right",this.open=e.open||!1,this.onClose=e.onClose||null,this.closeButton=e.closeButton!==!1,this.closeOnBackdrop=e.closeOnBackdrop!==!1,this.closeOnEscape=e.closeOnEscape!==!1}create(){let e=super.create("div",{className:j("fixed inset-0 z-50",this.open?"":"hidden")}),s=Se();e.appendChild(s),this.closeOnBackdrop&&s.addEventListener("click",()=>this.close());let o=this.position==="left"?"left-0":"right-0",h=document.createElement("div");if(h.className=j("absolute top-0 h-full w-96 bg-white shadow-lg transition-transform duration-300 flex flex-col z-50",o,this.open?"translate-x-0":this.position==="left"?"-translate-x-full":"translate-x-full"),this.title){let _=document.createElement("div");_.className="px-6 py-4 border-b border-gray-200 flex items-center justify-between";let L=document.createElement("h2");if(L.className="text-xl font-semibold text-gray-900",L.textContent=this.title,_.appendChild(L),this.closeButton){let A=document.createElement("button");A.className="text-gray-500 hover:text-gray-700 transition-colors duration-200",A.innerHTML="\xD7",A.addEventListener("click",()=>this.close()),_.appendChild(A)}h.appendChild(_)}let w=document.createElement("div");return w.className="flex-1 overflow-y-auto px-6 py-4",w.innerHTML=this.content,h.appendChild(w),e.appendChild(h),this.on("keydown",_=>{He.isEscape(_)&&this.closeOnEscape&&this.close()},{once:!1}),this.backdrop=s,this.drawerElement=h,e}open(){this.element||this.create(),this.open=!0,this.element.classList.remove("hidden"),this.drawerElement.classList.remove("-translate-x-full","translate-x-full"),this.drawerElement.classList.add("translate-x-0"),document.body.style.overflow="hidden"}close(){this.open=!1;let e=this.position==="left"?"-translate-x-full":"translate-x-full";this.drawerElement.classList.remove("translate-x-0"),this.drawerElement.classList.add(e),setTimeout(()=>{this.element?.parentNode&&this.element.classList.add("hidden"),document.body.style.overflow=""},300),this.onClose&&this.onClose()}toggle(){this.open?this.close():this.open()}}});var ps={};te(ps,{Empty:()=>nt});var nt,gs=ee(()=>{ie();oe();nt=class extends V{constructor(e={}){super(e),this.icon=e.icon||"\u{1F4E6}",this.title=e.title||"No data",this.message=e.message||"There is no data to display",this.action=e.action||null,this.className=e.className||""}create(){let e=super.create("div",{className:j("flex flex-col items-center justify-center p-8 text-center",this.className)}),s=O("div",{className:"text-6xl mb-4",text:this.icon});e.appendChild(s);let o=O("h3",{className:"text-lg font-semibold text-gray-900 mb-2",text:this.title});e.appendChild(o);let h=O("p",{className:"text-gray-500 mb-4",text:this.message});if(e.appendChild(h),this.action){let w=O("button",{className:"px-4 py-2 bg-blue-600 text-white rounded hover:bg-blue-700 transition-colors",text:this.action.label});w.addEventListener("click",this.action.onClick),e.appendChild(w)}return e}}});var hs={};te(hs,{Form:()=>ot});var ot,fs=ee(()=>{ie();oe();ot=class extends V{constructor(e={}){super(e),this.fields=e.fields||[],this.onSubmit=e.onSubmit||null,this.submitText=e.submitText||"Submit",this.className=e.className||""}create(){let e=super.create("form",{className:j("space-y-6",this.className)});e.addEventListener("submit",o=>{o.preventDefault(),this.handleSubmit()}),this.fieldElements={},this.fields.forEach(o=>{let h=O("div",{className:"space-y-2"});if(o.label){let L=O("label",{className:"block text-sm font-medium text-gray-700",text:o.label,attrs:{for:o.name}});h.appendChild(L)}let w=document.createElement(o.type==="textarea"?"textarea":"input");w.id=o.name,w.name=o.name,w.className=j("w-full px-3 py-2 border border-gray-300 rounded-md","focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent"),o.type!=="textarea"&&(w.type=o.type||"text"),o.placeholder&&(w.placeholder=o.placeholder),o.required&&(w.required=!0),o.disabled&&(w.disabled=!0),w.value=o.value||"",h.appendChild(w);let _=O("div",{className:"text-sm text-red-600 hidden",attrs:{"data-error":o.name}});h.appendChild(_),e.appendChild(h),this.fieldElements[o.name]=w});let s=O("button",{className:j("w-full px-4 py-2 bg-blue-600 text-white font-medium rounded","hover:bg-blue-700 transition-colors duration-200"),text:this.submitText,attrs:{type:"submit"}});return e.appendChild(s),e}handleSubmit(){let e={};Object.entries(this.fieldElements).forEach(([s,o])=>{e[s]=o.value}),this.onSubmit&&this.onSubmit(e)}getValues(){let e={};return Object.entries(this.fieldElements).forEach(([s,o])=>{e[s]=o.value}),e}setValues(e){Object.entries(e).forEach(([s,o])=>{this.fieldElements[s]&&(this.fieldElements[s].value=o)})}setError(e,s){let o=this.element.querySelector(`[data-error="${e}"]`);o&&(o.textContent=s,o.classList.remove("hidden"))}clearError(e){let s=this.element.querySelector(`[data-error="${e}"]`);s&&(s.textContent="",s.classList.add("hidden"))}reset(){this.element&&this.element.reset()}}});var ys={};te(ys,{HoverCard:()=>it});var it,bs=ee(()=>{ie();oe();it=class extends V{constructor(e={}){super(e),this.trigger=e.trigger||"Hover me",this.content=e.content||"",this.position=e.position||"bottom",this.delay=e.delay||200,this.className=e.className||"",this.visible=!1,this.timeoutId=null}create(){let e=super.create("div",{className:"relative inline-block"}),s=document.createElement("div");s.className="cursor-pointer px-3 py-2 rounded hover:bg-gray-100 transition-colors duration-200",s.textContent=this.trigger,e.appendChild(s);let o=document.createElement("div");return o.className=j("absolute hidden bg-white border border-gray-200 rounded-lg shadow-lg p-4 z-50","min-w-max max-w-sm",this.getPositionClasses(),this.className),o.innerHTML=this.content,e.appendChild(o),e.addEventListener("mouseenter",()=>this.show(o)),e.addEventListener("mouseleave",()=>this.hide(o)),this.cardElement=o,e}getPositionClasses(){let e={top:"bottom-full left-0 mb-2",bottom:"top-full left-0 mt-2",left:"right-full top-0 mr-2",right:"left-full top-0 ml-2"};return e[this.position]||e.bottom}show(e=this.cardElement){this.visible||!e||(this.timeoutId=setTimeout(()=>{this.visible=!0,e.classList.remove("hidden"),e.classList.add("opacity-100","transition-opacity","duration-200")},this.delay))}hide(e=this.cardElement){!this.visible||!e||(clearTimeout(this.timeoutId),this.visible=!1,e.classList.add("hidden"),e.classList.remove("opacity-100"))}setContent(e){this.content=e,this.cardElement&&(this.cardElement.innerHTML=e)}}});var xs={};te(xs,{Input:()=>lt});var lt,vs=ee(()=>{ie();oe();lt=class extends V{constructor(e={}){super(e),this.type=e.type||"text",this.placeholder=e.placeholder||"",this.value=e.value||"",this.name=e.name||"",this.disabled=e.disabled||!1,this.required=e.required||!1,this.className=e.className||"",this.onChange=e.onChange||null}create(){let s=super.create("input",{className:j("w-full px-3 py-2 border border-gray-300 rounded-md text-gray-900 placeholder-gray-500 transition-colors duration-200 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent disabled:bg-gray-100 disabled:cursor-not-allowed",this.className),attrs:{type:this.type,placeholder:this.placeholder,value:this.value,name:this.name,disabled:this.disabled?"":null,required:this.required?"":null}});return this.onChange&&(this.on("change",this.onChange),this.on("input",this.onChange)),s}getValue(){return this.element?.value||""}setValue(e){this.value=e,this.element&&(this.element.value=e)}setDisabled(e){this.disabled=e,this.attr("disabled",e?"":null)}setPlaceholder(e){this.placeholder=e,this.element&&(this.element.placeholder=e)}focus(){super.focus()}clear(){this.setValue("")}}});var ws={};te(ws,{InputGroup:()=>dt});var dt,ks=ee(()=>{ie();oe();dt=class extends V{constructor(e={}){super(e),this.prefix=e.prefix||null,this.suffix=e.suffix||null,this.input=e.input||null,this.className=e.className||""}create(){let s=super.create("div",{className:j("flex items-center border border-gray-300 rounded-md overflow-hidden focus-within:ring-2 focus-within:ring-blue-500",this.className)});if(this.prefix){let o=O("div",{className:"px-3 py-2 bg-gray-50 text-gray-700 font-medium text-sm",html:this.prefix});s.appendChild(o)}if(this.input){let o=this.input.element||this.input.create();o.classList.remove("border","focus:ring-2","focus:ring-blue-500"),s.appendChild(o)}if(this.suffix){let o=O("div",{className:"px-3 py-2 bg-gray-50 text-gray-700 font-medium text-sm",html:this.suffix});s.appendChild(o)}return s}}});var Es={};te(Es,{InputOTP:()=>ct});var ct,Cs=ee(()=>{ie();oe();ct=class extends V{constructor(e={}){super(e),this.length=e.length||6,this.value=e.value||"",this.className=e.className||"",this.onChange=e.onChange||null,this.onComplete=e.onComplete||null}create(){let e=super.create("div",{className:j("flex gap-2",this.className)});this.inputs=[];for(let s=0;s<this.length;s++){let o=document.createElement("input");o.type="text",o.maxLength="1",o.inputMode="numeric",o.className=j("w-12 h-12 text-center border-2 border-gray-300 rounded-lg font-semibold text-lg","focus:border-blue-600 focus:outline-none focus:ring-2 focus:ring-blue-200","transition-colors duration-200"),this.value&&this.value[s]&&(o.value=this.value[s]),o.addEventListener("input",h=>{let w=h.target.value;if(!/^\d*$/.test(w)){h.target.value="";return}w&&s<this.length-1&&this.inputs[s+1].focus(),this.updateValue()}),o.addEventListener("keydown",h=>{h.key==="Backspace"?!o.value&&s>0&&this.inputs[s-1].focus():h.key==="ArrowLeft"&&s>0?this.inputs[s-1].focus():h.key==="ArrowRight"&&s<this.length-1&&this.inputs[s+1].focus()}),this.inputs.push(o),e.appendChild(o)}return e}updateValue(){this.value=this.inputs.map(e=>e.value).join(""),this.onChange&&this.onChange(this.value),this.value.length===this.length&&this.onComplete&&this.onComplete(this.value)}getValue(){return this.value}setValue(e){this.value=e;for(let s=0;s<this.length;s++)this.inputs[s].value=e[s]||""}clear(){this.inputs.forEach(e=>{e.value=""}),this.value=""}focus(){this.inputs.length>0&&this.inputs[0].focus()}}});var Ls={};te(Ls,{Item:()=>ut});var ut,_s=ee(()=>{ie();oe();ut=class extends V{constructor(e={}){super(e),this.label=e.label||"",this.value=e.value||"",this.icon=e.icon||null,this.className=e.className||"",this.selected=e.selected||!1,this.disabled=e.disabled||!1}create(){let e="flex items-center gap-2 px-3 py-2 rounded cursor-pointer transition-colors duration-200 hover:bg-gray-100 disabled:opacity-50 disabled:cursor-not-allowed",s=this.selected?"bg-blue-50 text-blue-600":"text-gray-900",o=super.create("div",{className:j(e,s,this.className),attrs:{role:"option","aria-selected":this.selected,disabled:this.disabled?"":null,"data-value":this.value}}),h="";return this.icon&&(h+=`<span class="flex-shrink-0">${this.icon}</span>`),h+=`<span>${this.label}</span>`,o.innerHTML=h,o}setSelected(e){this.selected=e,this.element&&(this.attr("aria-selected",e),this.toggleClass("bg-blue-50 text-blue-600",e))}setDisabled(e){this.disabled=e,this.attr("disabled",e?"":null)}}});var $s={};te($s,{Label:()=>mt});var mt,Ts=ee(()=>{ie();oe();mt=class extends V{constructor(e={}){super(e),this.text=e.text||"",this.htmlFor=e.htmlFor||"",this.required=e.required||!1,this.className=e.className||""}create(){let s=super.create("label",{className:j("block text-sm font-medium text-gray-700 mb-1",this.className),attrs:{for:this.htmlFor}}),o=this.text;return this.required&&(o+=' <span class="text-red-500 ml-1">*</span>'),s.innerHTML=o,s}setText(e){if(this.text=e,this.element){let s=e;this.required&&(s+=' <span class="text-red-500 ml-1">*</span>'),this.element.innerHTML=s}}setRequired(e){if(this.required=e,this.element){let s=this.element.querySelector('[class*="text-red"]');e&&!s?this.element.innerHTML+=' <span class="text-red-500 ml-1">*</span>':!e&&s&&s.remove()}}}});var Is={};te(Is,{Kbd:()=>pt});var pt,As=ee(()=>{ie();oe();pt=class extends V{constructor(e={}){super(e),this.label=e.label||"K",this.className=e.className||""}create(){let s=super.create("kbd",{className:j("px-2 py-1 bg-gray-100 border border-gray-300 rounded text-xs font-semibold text-gray-900 inline-block font-mono",this.className)});return s.textContent=this.label,s}setLabel(e){this.label=e,this.element&&(this.element.textContent=e)}}});var Ss={};te(Ss,{NativeSelect:()=>gt});var gt,Bs=ee(()=>{ie();oe();gt=class extends V{constructor(e={}){super(e),this.items=e.items||[],this.selected=e.selected||"",this.placeholder=e.placeholder||"Select...",this.disabled=e.disabled||!1,this.required=e.required||!1,this.name=e.name||"",this.className=e.className||"",this.onChange=e.onChange||null}create(){let s=super.create("select",{className:j("w-full px-3 py-2 border border-gray-300 rounded-md text-gray-900 transition-colors duration-200 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent disabled:bg-gray-100 disabled:cursor-not-allowed bg-white appearance-none cursor-pointer",this.className),attrs:{disabled:this.disabled?"":null,required:this.required?"":null,...this.name&&{name:this.name}}});if(this.placeholder){let o=document.createElement("option");o.value="",o.textContent=this.placeholder,o.disabled=!0,s.appendChild(o)}return this.items.forEach(o=>{let h=document.createElement("option");h.value=o.value,h.textContent=o.label,o.value===this.selected&&(h.selected=!0),s.appendChild(h)}),this.onChange&&this.on("change",this.onChange),s}getValue(){return this.element?.value||""}setValue(e){this.selected=e,this.element&&(this.element.value=e)}setDisabled(e){this.disabled=e,this.attr("disabled",e?"":null)}addItem(e,s){if(this.element){let o=document.createElement("option");o.value=s,o.textContent=e,this.element.appendChild(o)}}removeItem(e){if(this.element){let s=this.element.querySelector(`option[value="${e}"]`);s&&s.remove()}}}});var Ms={};te(Ms,{Tooltip:()=>ht});var ht,qs=ee(()=>{ie();oe();ht=class extends V{constructor(e={}){super(e),this.content=e.content||"",this.position=e.position||"top",this.delay=e.delay||200,this.trigger=e.trigger||"hover",this.className=e.className||"",this.visible=!1,this.timeoutId=null}create(){let e=super.create("div",{className:"relative inline-block"}),s=document.createElement("div");s.className=j("absolute hidden bg-gray-900 text-white px-3 py-2 rounded text-sm whitespace-nowrap z-50","opacity-0 transition-opacity duration-200",this.getPositionClasses(),this.className),s.textContent=this.content;let o=document.createElement("div");return o.className=j("absolute w-2 h-2 bg-gray-900 transform rotate-45",this.getArrowClasses()),s.appendChild(o),e.appendChild(s),this.tooltipElement=s,this.trigger==="hover"?(e.addEventListener("mouseenter",()=>this.show()),e.addEventListener("mouseleave",()=>this.hide())):this.trigger==="focus"&&(e.addEventListener("focus",()=>this.show(),!0),e.addEventListener("blur",()=>this.hide(),!0)),e}getPositionClasses(){let e={top:"bottom-full left-1/2 transform -translate-x-1/2 mb-2",bottom:"top-full left-1/2 transform -translate-x-1/2 mt-2",left:"right-full top-1/2 transform -translate-y-1/2 mr-2",right:"left-full top-1/2 transform -translate-y-1/2 ml-2"};return e[this.position]||e.top}getArrowClasses(){let e={top:"top-full left-1/2 transform -translate-x-1/2 -translate-y-1/2",bottom:"bottom-full left-1/2 transform -translate-x-1/2 translate-y-1/2",left:"left-full top-1/2 transform translate-x-1/2 -translate-y-1/2",right:"right-full top-1/2 transform -translate-x-1/2 -translate-y-1/2"};return e[this.position]||e.top}show(){this.visible||(this.timeoutId=setTimeout(()=>{this.visible=!0,this.tooltipElement.classList.remove("hidden"),this.tooltipElement.classList.add("opacity-100")},this.delay))}hide(){this.visible&&(clearTimeout(this.timeoutId),this.visible=!1,this.tooltipElement.classList.remove("opacity-100"),this.tooltipElement.classList.add("hidden"))}setContent(e){this.content=e,this.tooltipElement&&(this.tooltipElement.textContent=e)}}});var Ps={};te(Ps,{Toggle:()=>ft});var ft,Hs=ee(()=>{ie();oe();ft=class extends V{constructor(e={}){super(e),this.label=e.label||"",this.pressed=e.pressed||!1,this.disabled=e.disabled||!1,this.variant=e.variant||"default",this.size=e.size||"md",this.className=e.className||"",this.onChange=e.onChange||null}create(){let e="px-4 py-2 font-medium rounded transition-all duration-200 focus:outline-none focus:ring-2 focus:ring-offset-2 disabled:opacity-50 disabled:cursor-not-allowed",s={default:this.pressed?"bg-gray-900 text-white":"bg-gray-100 text-gray-900 hover:bg-gray-200",outline:this.pressed?"border-2 border-gray-900 bg-gray-900 text-white":"border-2 border-gray-300 text-gray-900 hover:bg-gray-50"},o={sm:"px-2 py-1 text-sm",md:"px-4 py-2 text-base",lg:"px-6 py-3 text-lg"},h=super.create("button",{className:j(e,s[this.variant],o[this.size],this.className),attrs:{"aria-pressed":this.pressed,disabled:this.disabled?"":null}});return h.textContent=this.label,this.on("click",()=>{this.toggle()}),h}isPressed(){return this.pressed}setPressed(e){this.pressed=e,this.element&&(this.attr("aria-pressed",e),this.toggleClass("bg-gray-900 text-white",e),this.onChange&&this.onChange(e))}toggle(){this.setPressed(!this.pressed)}setDisabled(e){this.disabled=e,this.attr("disabled",e?"":null)}}});var js={};te(js,{ToggleGroup:()=>yt});var yt,Ns=ee(()=>{ie();oe();yt=class extends V{constructor(e={}){super(e),this.items=e.items||[],this.selected=e.selected||null,this.multiple=e.multiple||!1,this.orientation=e.orientation||"horizontal",this.className=e.className||"",this.onChange=e.onChange||null}create(){let e=this.orientation==="vertical"?"flex-col":"flex-row",o=super.create("div",{className:j("flex",e,"inline-flex border border-gray-300 rounded-md overflow-hidden",this.className),attrs:{role:"group"}});return this.toggleButtons=[],this.items.forEach((h,w)=>{let _=this.multiple?Array.isArray(this.selected)&&this.selected.includes(h.value):h.value===this.selected,L=j("flex-1 px-4 py-2 font-medium transition-colors duration-200",_?"bg-blue-600 text-white":"bg-white text-gray-700 hover:bg-gray-50",w>0?this.orientation==="vertical"?"border-t border-gray-300":"border-l border-gray-300":""),A=O("button",{className:L,text:h.label,attrs:{"data-value":h.value,"aria-pressed":_,type:"button"}});A.addEventListener("click",()=>{if(this.multiple){Array.isArray(this.selected)||(this.selected=[]);let Y=this.selected.indexOf(h.value);Y>-1?this.selected.splice(Y,1):this.selected.push(h.value)}else this.selected=h.value;this.updateView(),this.onChange&&this.onChange(this.selected)}),o.appendChild(A),this.toggleButtons.push(A)}),o}updateView(){this.toggleButtons.forEach(e=>{let s=e.getAttribute("data-value"),o=this.multiple?Array.isArray(this.selected)&&this.selected.includes(s):s===this.selected;e.setAttribute("aria-pressed",o),e.className=j("flex-1 px-4 py-2 font-medium transition-colors duration-200",o?"bg-blue-600 text-white":"bg-white text-gray-700 hover:bg-gray-50")})}getValue(){return this.selected}setValue(e){this.selected=e,this.updateView()}}});var Fs={};te(Fs,{Toast:()=>bt});var bt,zs=ee(()=>{ie();oe();bt=class n extends V{constructor(e={}){super(e),this.message=e.message||"",this.type=e.type||"default",typeof e.duration<"u"?this.duration=e.duration:this.duration=typeof window<"u"&&window.innerWidth<640?2500:3e3,this.position=e.position||"top-right",this.className=e.className||"",this.onClose=e.onClose||null}destroy(){if(!this.element)return;let e=this.element;e.classList.add(this.getExitAnimationClass());let s=()=>{e.removeEventListener("animationend",s),super.destroy();let o=n.getContainer(this.position);o&&o.childElementCount===0&&o.parentNode&&(o.parentNode.removeChild(o),n._containers&&delete n._containers[this.position||"top-right"])};e.addEventListener("animationend",s),setTimeout(s,320)}static getContainer(e){let s=e||"top-right";if(this._containers||(this._containers={}),this._containers[s]&&document.body.contains(this._containers[s]))return this._containers[s];let o=O("div",{className:j("fixed z-50 p-2 flex flex-col gap-2 pointer-events-none",this.getPositionClassesForContainer(s))});return document.body.appendChild(o),this._containers[s]=o,o}static getPositionClassesForContainer(e){switch(e){case"top-left":return"top-4 left-4 items-start";case"top-right":return"top-4 right-4 items-end";case"bottom-left":return"bottom-4 left-4 items-start";case"bottom-right":return"bottom-4 right-4 items-end";case"top-center":return"top-4 left-1/2 -translate-x-1/2 items-center transform";default:return"top-4 right-4 items-end"}}create(){let e=n.getContainer(this.position),s=O("div",{className:j("rounded-lg shadow-lg p-2.5 flex items-center gap-2 min-w-0 max-w-[90vw] sm:max-w-sm bg-opacity-95",this.getEnterAnimationClass(),this.getTypeClasses(),this.className)}),o=O("span",{className:"text-base flex-shrink-0",text:this.getIcon()});s.appendChild(o);let h=O("span",{text:this.message,className:"flex-1 text-sm sm:text-base"});s.appendChild(h),s.setAttribute("role",this.type==="error"?"alert":"status"),s.setAttribute("aria-live",this.type==="error"?"assertive":"polite");let w=O("button",{className:"text-base hover:opacity-70 transition-opacity flex-shrink-0",text:"\xD7"});for(w.setAttribute("aria-label","Dismiss notification"),w.addEventListener("click",()=>{this.destroy()}),s.appendChild(w),this.element=s,e.appendChild(this.element);e.children.length>3;)e.removeChild(e.firstElementChild);return this.duration>0&&setTimeout(()=>{this.destroy()},this.duration),this.element}getEnterAnimationClass(){let e=this.position||"top-right";return e==="top-right"||e==="bottom-right"?"animate-slide-in-right transition-all duration-300 pointer-events-auto":e==="top-left"||e==="bottom-left"?"animate-slide-in-left transition-all duration-300 pointer-events-auto":"animate-slide-in-top transition-all duration-300 pointer-events-auto"}getExitAnimationClass(){let e=this.position||"top-right";return e==="top-right"||e==="bottom-right"?"animate-slide-out-right":e==="top-left"||e==="bottom-left"?"animate-slide-out-left":"animate-slide-out-top"}getPositionClasses(){let e={"top-left":"top-4 left-4","top-right":"top-4 right-4","bottom-left":"bottom-4 left-4","bottom-right":"bottom-4 right-4","top-center":"top-4 left-1/2 -translate-x-1/2 transform"};return e[this.position]||e["bottom-right"]}getTypeClasses(){let e={default:"bg-gray-900 text-white",success:"bg-green-600 text-white",error:"bg-red-600 text-white",warning:"bg-yellow-600 text-white",info:"bg-blue-600 text-white"};return e[this.type]||e.default}getIcon(){let e={default:"\u2139",success:"\u2713",error:"\u2715",warning:"\u26A0",info:"\u2139"};return e[this.type]||e.default}static show(e,s={}){let o=new n({message:e,...s}),h=o.create();return o}static success(e,s={}){return this.show(e,{...s,type:"success"})}static error(e,s={}){return this.show(e,{...s,type:"error",position:s.position||"top-right"})}static info(e,s={}){return this.show(e,{...s,type:"info"})}static warning(e,s={}){return this.show(e,{...s,type:"warning"})}};if(!document.querySelector("style[data-toast]")){let n=document.createElement("style");n.setAttribute("data-toast","true"),n.textContent=`
    @keyframes slideInTop {
      from { transform: translateY(-12px); opacity: 0; }
      to { transform: translateY(0); opacity: 1; }
    }
    .animate-slide-in-top { animation: slideInTop 0.25s ease-out; }

    @keyframes slideOutTop {
      from { transform: translateY(0); opacity: 1; }
      to { transform: translateY(-12px); opacity: 0; }
    }
    .animate-slide-out-top { animation: slideOutTop 0.2s ease-in forwards; }

    @keyframes slideInRight {
      from { transform: translateX(16px); opacity: 0; }
      to { transform: translateX(0); opacity: 1; }
    }
    .animate-slide-in-right { animation: slideInRight 0.25s ease-out; }

    @keyframes slideOutRight {
      from { transform: translateX(0); opacity: 1; }
      to { transform: translateX(16px); opacity: 0; }
    }
    .animate-slide-out-right { animation: slideOutRight 0.2s ease-in forwards; }

    @keyframes slideInLeft {
      from { transform: translateX(-16px); opacity: 0; }
      to { transform: translateX(0); opacity: 1; }
    }
    .animate-slide-in-left { animation: slideInLeft 0.25s ease-out; }

    @keyframes slideOutLeft {
      from { transform: translateX(0); opacity: 1; }
      to { transform: translateX(-16px); opacity: 0; }
    }
    .animate-slide-out-left { animation: slideOutLeft 0.2s ease-in forwards; }
  `,document.head.appendChild(n)}});function Rs({root:n=document,selector:e="[data-hydrate]",threshold:s=.15}={}){if(typeof IntersectionObserver>"u")return;let o=new IntersectionObserver(async(h,w)=>{for(let _ of h){if(!_.isIntersecting)continue;let L=_.target,A=L.dataset.hydrate||"",[Y,Z="init"]=A.split("#");if(!Y){w.unobserve(L);continue}try{let z=null,E=Hr[Y];if(typeof E=="function")z=await E();else throw new Error("Module not registered for lazy hydration: "+Y);let H=z[Z]||z.default||null;if(typeof H=="function")try{H(L)}catch(U){console.error("hydrate init failed",U)}}catch(z){console.error("lazy hydrate import failed for",Y,z)}finally{w.unobserve(L)}}},{threshold:s});n.querySelectorAll(e).forEach(h=>o.observe(h))}var Hr,Os=ee(()=>{Hr={"components/Alert.js":()=>Promise.resolve().then(()=>(At(),It)),"components/AlertDialog.js":()=>Promise.resolve().then(()=>(Bt(),St)),"components/Avatar.js":()=>Promise.resolve().then(()=>(qt(),Mt)),"components/Badge.js":()=>Promise.resolve().then(()=>(Ht(),Pt)),"components/Button.js":()=>Promise.resolve().then(()=>(Nt(),jt)),"components/ButtonGroup.js":()=>Promise.resolve().then(()=>(zt(),Ft)),"components/Breadcrumb.js":()=>Promise.resolve().then(()=>(Ot(),Rt)),"components/Card.js":()=>Promise.resolve().then(()=>(Dt(),Vt)),"components/Carousel.js":()=>Promise.resolve().then(()=>(Wt(),Ut)),"components/Chart.js":()=>Promise.resolve().then(()=>(Gt(),Yt)),"components/Checkbox.js":()=>Promise.resolve().then(()=>(Jt(),Qt)),"components/Collapsible.js":()=>Promise.resolve().then(()=>(Kt(),Xt)),"components/Command.js":()=>Promise.resolve().then(()=>(es(),Zt)),"components/Combobox.js":()=>Promise.resolve().then(()=>(ss(),ts)),"components/ContextMenu.js":()=>Promise.resolve().then(()=>(as(),rs)),"components/DatePicker.js":()=>Promise.resolve().then(()=>(os(),ns)),"components/Dialog.js":()=>Promise.resolve().then(()=>(ls(),is)),"components/DropdownMenu.js":()=>Promise.resolve().then(()=>(cs(),ds)),"components/Drawer.js":()=>Promise.resolve().then(()=>(ms(),us)),"components/Empty.js":()=>Promise.resolve().then(()=>(gs(),ps)),"components/Form.js":()=>Promise.resolve().then(()=>(fs(),hs)),"components/HoverCard.js":()=>Promise.resolve().then(()=>(bs(),ys)),"components/Input.js":()=>Promise.resolve().then(()=>(vs(),xs)),"components/InputGroup.js":()=>Promise.resolve().then(()=>(ks(),ws)),"components/InputOTP.js":()=>Promise.resolve().then(()=>(Cs(),Es)),"components/Item.js":()=>Promise.resolve().then(()=>(_s(),Ls)),"components/Label.js":()=>Promise.resolve().then(()=>(Ts(),$s)),"components/Kbd.js":()=>Promise.resolve().then(()=>(As(),Is)),"components/NativeSelect.js":()=>Promise.resolve().then(()=>(Bs(),Ss)),"components/Tooltip.js":()=>Promise.resolve().then(()=>(qs(),Ms)),"components/Toggle.js":()=>Promise.resolve().then(()=>(Hs(),Ps)),"components/ToggleGroup.js":()=>Promise.resolve().then(()=>(Ns(),js)),"components/Toast.js":()=>Promise.resolve().then(()=>(zs(),Fs))}});var Ds={};te(Ds,{default:()=>jr});var Vs,jr,Us=ee(()=>{Vs=(function(){"use strict";let n=null;async function e(){if(AuthGuard.protectPage()){await Y();try{U()}catch{}Z(),H(),B(),C(),s()}}function s(){o(),h(),w(),_(),L(),A()}function o(){let i=document.getElementById("loyalty-points");if(!i)return;let a=n?.loyalty_points||Math.floor(Math.random()*500)+100,c=a>=500?"Gold":a>=200?"Silver":"Bronze",u={Bronze:"from-amber-600 to-amber-700",Silver:"from-gray-400 to-gray-500",Gold:"from-yellow-400 to-yellow-500"},m=c==="Gold"?null:c==="Silver"?"Gold":"Silver",b=c==="Gold"?0:c==="Silver"?500:200,v=m?Math.min(100,a/b*100):100;i.innerHTML=`
            <div class="bg-gradient-to-br ${u[c]} rounded-2xl p-6 text-white relative overflow-hidden">
                <div class="absolute top-0 right-0 w-32 h-32 opacity-10">
                    <svg viewBox="0 0 24 24" fill="currentColor" class="w-full h-full">
                        <path d="M12 2l3.09 6.26L22 9.27l-5 4.87 1.18 6.88L12 17.77l-6.18 3.25L7 14.14 2 9.27l6.91-1.01L12 2z"/>
                    </svg>
                </div>
                <div class="relative">
                    <div class="flex items-center justify-between mb-4">
                        <div>
                            <p class="text-white/80 text-sm font-medium">${c} Member</p>
                            <p class="text-3xl font-bold">${a.toLocaleString()} pts</p>
                        </div>
                        <div class="w-14 h-14 bg-white/20 backdrop-blur rounded-xl flex items-center justify-center">
                            <svg class="w-8 h-8" fill="currentColor" viewBox="0 0 24 24">
                                <path d="M12 2l3.09 6.26L22 9.27l-5 4.87 1.18 6.88L12 17.77l-6.18 3.25L7 14.14 2 9.27l6.91-1.01L12 2z"/>
                            </svg>
                        </div>
                    </div>
                    ${m?`
                        <div class="mt-4">
                            <div class="flex justify-between text-sm mb-1">
                                <span>${b-a} points to ${m}</span>
                                <span>${Math.round(v)}%</span>
                            </div>
                            <div class="w-full bg-white/30 rounded-full h-2">
                                <div class="bg-white h-2 rounded-full transition-all duration-500" style="width: ${v}%"></div>
                            </div>
                        </div>
                    `:`<p class="text-sm text-white/80 mt-2">\u{1F389} You've reached the highest tier!</p>`}
                    <div class="mt-4 pt-4 border-t border-white/20">
                        <a href="/loyalty/" class="text-sm font-medium hover:underline flex items-center gap-1">
                            View Rewards
                            <svg class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9 5l7 7-7 7"/></svg>
                        </a>
                    </div>
                </div>
            </div>
        `}function h(){let i=document.getElementById("quick-stats");if(!i)return;let a={totalOrders:n?.total_orders||Math.floor(Math.random()*20)+5,totalSpent:n?.total_spent||Math.floor(Math.random()*1e3)+200,wishlistItems:n?.wishlist_count||Math.floor(Math.random()*10)+2,savedAddresses:n?.address_count||Math.floor(Math.random()*3)+1};i.innerHTML=`
            <div class="grid grid-cols-2 md:grid-cols-4 gap-4">
                <div class="bg-white dark:bg-stone-800 rounded-xl p-4 border border-gray-100 dark:border-stone-700">
                    <div class="flex items-center gap-3">
                        <div class="w-10 h-10 bg-blue-100 dark:bg-blue-900/30 rounded-lg flex items-center justify-center">
                            <svg class="w-5 h-5 text-blue-600 dark:text-blue-400" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9 5H7a2 2 0 00-2 2v12a2 2 0 002 2h10a2 2 0 002-2V7a2 2 0 00-2-2h-2M9 5a2 2 0 002 2h2a2 2 0 002-2M9 5a2 2 0 012-2h2a2 2 0 012 2"/></svg>
                        </div>
                        <div>
                            <p class="text-2xl font-bold text-stone-900 dark:text-white">${a.totalOrders}</p>
                            <p class="text-xs text-stone-500 dark:text-stone-400">Total Orders</p>
                        </div>
                    </div>
                </div>
                <div class="bg-white dark:bg-stone-800 rounded-xl p-4 border border-gray-100 dark:border-stone-700">
                    <div class="flex items-center gap-3">
                        <div class="w-10 h-10 bg-green-100 dark:bg-green-900/30 rounded-lg flex items-center justify-center">
                            <svg class="w-5 h-5 text-green-600 dark:text-green-400" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 8c-1.657 0-3 .895-3 2s1.343 2 3 2 3 .895 3 2-1.343 2-3 2m0-8c1.11 0 2.08.402 2.599 1M12 8V7m0 1v8m0 0v1m0-1c-1.11 0-2.08-.402-2.599-1M21 12a9 9 0 11-18 0 9 9 0 0118 0z"/></svg>
                        </div>
                        <div>
                            <p class="text-2xl font-bold text-stone-900 dark:text-white">${Templates.formatPrice(a.totalSpent)}</p>
                            <p class="text-xs text-stone-500 dark:text-stone-400">Total Spent</p>
                        </div>
                    </div>
                </div>
                <div class="bg-white dark:bg-stone-800 rounded-xl p-4 border border-gray-100 dark:border-stone-700">
                    <div class="flex items-center gap-3">
                        <div class="w-10 h-10 bg-rose-100 dark:bg-rose-900/30 rounded-lg flex items-center justify-center">
                            <svg class="w-5 h-5 text-rose-600 dark:text-rose-400" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M4.318 6.318a4.5 4.5 0 000 6.364L12 20.364l7.682-7.682a4.5 4.5 0 00-6.364-6.364L12 7.636l-1.318-1.318a4.5 4.5 0 00-6.364 0z"/></svg>
                        </div>
                        <div>
                            <p class="text-2xl font-bold text-stone-900 dark:text-white">${a.wishlistItems}</p>
                            <p class="text-xs text-stone-500 dark:text-stone-400">Wishlist Items</p>
                        </div>
                    </div>
                </div>
                <div class="bg-white dark:bg-stone-800 rounded-xl p-4 border border-gray-100 dark:border-stone-700">
                    <div class="flex items-center gap-3">
                        <div class="w-10 h-10 bg-purple-100 dark:bg-purple-900/30 rounded-lg flex items-center justify-center">
                            <svg class="w-5 h-5 text-purple-600 dark:text-purple-400" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M17.657 16.657L13.414 20.9a1.998 1.998 0 01-2.827 0l-4.244-4.243a8 8 0 1111.314 0z"/><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M15 11a3 3 0 11-6 0 3 3 0 016 0z"/></svg>
                        </div>
                        <div>
                            <p class="text-2xl font-bold text-stone-900 dark:text-white">${a.savedAddresses}</p>
                            <p class="text-xs text-stone-500 dark:text-stone-400">Saved Addresses</p>
                        </div>
                    </div>
                </div>
            </div>
        `}async function w(){let i=document.getElementById("recent-activity");if(i)try{let c=(await OrdersApi.getAll({limit:3})).data||[];if(c.length===0){i.innerHTML=`
                    <div class="text-center py-6 text-stone-500 dark:text-stone-400">
                        <p>No recent activity</p>
                    </div>
                `;return}i.innerHTML=`
                <div class="bg-white dark:bg-stone-800 rounded-xl border border-gray-100 dark:border-stone-700 overflow-hidden">
                    <div class="px-4 py-3 border-b border-gray-100 dark:border-stone-700 flex items-center justify-between">
                        <h3 class="font-semibold text-stone-900 dark:text-white">Recent Orders</h3>
                        <a href="/orders/" class="text-sm text-primary-600 dark:text-amber-400 hover:underline">View All</a>
                    </div>
                    <div class="divide-y divide-gray-100 dark:divide-stone-700">
                        ${c.map(u=>{let b={pending:"text-yellow-600 dark:text-yellow-400",processing:"text-blue-600 dark:text-blue-400",shipped:"text-indigo-600 dark:text-indigo-400",delivered:"text-green-600 dark:text-green-400",cancelled:"text-red-600 dark:text-red-400"}[u.status]||"text-stone-600 dark:text-stone-400",v=u.items?.[0];return`
                                <a href="/orders/${u.id}/" class="flex items-center gap-4 p-4 hover:bg-stone-50 dark:hover:bg-stone-700/50 transition-colors">
                                    <div class="w-12 h-12 bg-stone-100 dark:bg-stone-700 rounded-lg overflow-hidden flex-shrink-0">
                                        ${v?.product?.image?`<img src="${v.product.image}" alt="" class="w-full h-full object-cover">`:`<div class="w-full h-full flex items-center justify-center text-stone-400">
                                                <svg class="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="1.5" d="M20 7l-8-4-8 4m16 0l-8 4m8-4v10l-8 4m0-10L4 7m8 4v10M4 7v10l8 4"/></svg>
                                            </div>`}
                                    </div>
                                    <div class="flex-1 min-w-0">
                                        <p class="font-medium text-stone-900 dark:text-white truncate">Order #${Templates.escapeHtml(u.order_number||u.id)}</p>
                                        <p class="text-sm ${b}">${Templates.escapeHtml(u.status_display||u.status)}</p>
                                    </div>
                                    <div class="text-right">
                                        <p class="font-semibold text-stone-900 dark:text-white">${Templates.formatPrice(u.total)}</p>
                                        <p class="text-xs text-stone-500 dark:text-stone-400">${Templates.formatDate(u.created_at)}</p>
                                    </div>
                                </a>
                            `}).join("")}
                    </div>
                </div>
            `}catch{i.innerHTML=""}}function _(){let i=document.getElementById("notification-preferences");if(!i)return;let a=JSON.parse(localStorage.getItem("notificationPreferences")||"{}"),u={...{orderUpdates:!0,promotions:!0,newArrivals:!1,priceDrops:!0,newsletter:!1},...a};i.innerHTML=`
            <div class="space-y-4">
                <div class="flex items-center justify-between">
                    <div>
                        <p class="font-medium text-stone-900 dark:text-white">Order Updates</p>
                        <p class="text-sm text-stone-500 dark:text-stone-400">Get notified about your order status</p>
                    </div>
                    <label class="relative inline-flex items-center cursor-pointer">
                        <input type="checkbox" class="sr-only peer" data-pref="orderUpdates" ${u.orderUpdates?"checked":""}>
                        <div class="w-11 h-6 bg-stone-200 dark:bg-stone-600 peer-focus:ring-2 peer-focus:ring-primary-300 dark:peer-focus:ring-amber-800 rounded-full peer peer-checked:after:translate-x-full peer-checked:after:border-white after:content-[''] after:absolute after:top-[2px] after:left-[2px] after:bg-white after:rounded-full after:h-5 after:w-5 after:transition-all peer-checked:bg-primary-600 dark:peer-checked:bg-amber-500"></div>
                    </label>
                </div>
                <div class="flex items-center justify-between">
                    <div>
                        <p class="font-medium text-stone-900 dark:text-white">Promotions & Sales</p>
                        <p class="text-sm text-stone-500 dark:text-stone-400">Be the first to know about deals</p>
                    </div>
                    <label class="relative inline-flex items-center cursor-pointer">
                        <input type="checkbox" class="sr-only peer" data-pref="promotions" ${u.promotions?"checked":""}>
                        <div class="w-11 h-6 bg-stone-200 dark:bg-stone-600 peer-focus:ring-2 peer-focus:ring-primary-300 dark:peer-focus:ring-amber-800 rounded-full peer peer-checked:after:translate-x-full peer-checked:after:border-white after:content-[''] after:absolute after:top-[2px] after:left-[2px] after:bg-white after:rounded-full after:h-5 after:w-5 after:transition-all peer-checked:bg-primary-600 dark:peer-checked:bg-amber-500"></div>
                    </label>
                </div>
                <div class="flex items-center justify-between">
                    <div>
                        <p class="font-medium text-stone-900 dark:text-white">Price Drops</p>
                        <p class="text-sm text-stone-500 dark:text-stone-400">Alert when wishlist items go on sale</p>
                    </div>
                    <label class="relative inline-flex items-center cursor-pointer">
                        <input type="checkbox" class="sr-only peer" data-pref="priceDrops" ${u.priceDrops?"checked":""}>
                        <div class="w-11 h-6 bg-stone-200 dark:bg-stone-600 peer-focus:ring-2 peer-focus:ring-primary-300 dark:peer-focus:ring-amber-800 rounded-full peer peer-checked:after:translate-x-full peer-checked:after:border-white after:content-[''] after:absolute after:top-[2px] after:left-[2px] after:bg-white after:rounded-full after:h-5 after:w-5 after:transition-all peer-checked:bg-primary-600 dark:peer-checked:bg-amber-500"></div>
                    </label>
                </div>
                <div class="flex items-center justify-between">
                    <div>
                        <p class="font-medium text-stone-900 dark:text-white">New Arrivals</p>
                        <p class="text-sm text-stone-500 dark:text-stone-400">Updates on new products</p>
                    </div>
                    <label class="relative inline-flex items-center cursor-pointer">
                        <input type="checkbox" class="sr-only peer" data-pref="newArrivals" ${u.newArrivals?"checked":""}>
                        <div class="w-11 h-6 bg-stone-200 dark:bg-stone-600 peer-focus:ring-2 peer-focus:ring-primary-300 dark:peer-focus:ring-amber-800 rounded-full peer peer-checked:after:translate-x-full peer-checked:after:border-white after:content-[''] after:absolute after:top-[2px] after:left-[2px] after:bg-white after:rounded-full after:h-5 after:w-5 after:transition-all peer-checked:bg-primary-600 dark:peer-checked:bg-amber-500"></div>
                    </label>
                </div>
            </div>
        `,i.querySelectorAll("input[data-pref]").forEach(m=>{m.addEventListener("change",()=>{let b=m.dataset.pref,v=JSON.parse(localStorage.getItem("notificationPreferences")||"{}");v[b]=m.checked,localStorage.setItem("notificationPreferences",JSON.stringify(v)),Toast.success("Preference saved")})})}function L(){let i=document.getElementById("quick-reorder");if(!i)return;let a=JSON.parse(localStorage.getItem("recentlyOrdered")||"[]");if(a.length===0){i.classList.add("hidden");return}i.innerHTML=`
            <div class="bg-white dark:bg-stone-800 rounded-xl border border-gray-100 dark:border-stone-700 p-4">
                <h3 class="font-semibold text-stone-900 dark:text-white mb-4">Quick Reorder</h3>
                <div class="flex gap-3 overflow-x-auto pb-2">
                    ${a.slice(0,5).map(c=>`
                        <button class="quick-reorder-btn flex-shrink-0 flex flex-col items-center gap-2 p-3 bg-stone-50 dark:bg-stone-700 rounded-xl hover:bg-stone-100 dark:hover:bg-stone-600 transition-colors" data-product-id="${c.id}">
                            <div class="w-16 h-16 rounded-lg bg-stone-200 dark:bg-stone-600 overflow-hidden">
                                <img src="${c.image||"/static/images/placeholder.jpg"}" alt="${Templates.escapeHtml(c.name)}" class="w-full h-full object-cover">
                            </div>
                            <span class="text-xs font-medium text-stone-700 dark:text-stone-300 text-center line-clamp-2 w-20">${Templates.escapeHtml(c.name)}</span>
                        </button>
                    `).join("")}
                </div>
            </div>
        `,i.querySelectorAll(".quick-reorder-btn").forEach(c=>{c.addEventListener("click",async()=>{let u=c.dataset.productId;c.disabled=!0;try{await CartApi.addItem(u,1),Toast.success("Added to cart!"),document.dispatchEvent(new CustomEvent("cart:updated"))}catch{Toast.error("Failed to add to cart")}finally{c.disabled=!1}})})}function A(){let i=document.getElementById("security-check");if(!i)return;let a=0,c=[],u=n?.email_verified!==!1;u&&(a+=25),c.push({label:"Email verified",completed:u});let m=!!n?.phone;m&&(a+=25),c.push({label:"Phone number added",completed:m});let b=n?.two_factor_enabled||!1;b&&(a+=25),c.push({label:"Two-factor authentication",completed:b});let v=!0;v&&(a+=25),c.push({label:"Strong password",completed:v});let P=a>=75?"text-green-600 dark:text-green-400":a>=50?"text-yellow-600 dark:text-yellow-400":"text-red-600 dark:text-red-400",D=a>=75?"bg-green-500":a>=50?"bg-yellow-500":"bg-red-500";i.innerHTML=`
            <div class="bg-white dark:bg-stone-800 rounded-xl border border-gray-100 dark:border-stone-700 p-4">
                <div class="flex items-center justify-between mb-4">
                    <h3 class="font-semibold text-stone-900 dark:text-white">Account Security</h3>
                    <span class="${P} font-bold">${a}%</span>
                </div>
                <div class="w-full bg-stone-200 dark:bg-stone-600 rounded-full h-2 mb-4">
                    <div class="${D} h-2 rounded-full transition-all duration-500" style="width: ${a}%"></div>
                </div>
                <div class="space-y-2">
                    ${c.map(J=>`
                        <div class="flex items-center gap-2 text-sm">
                            ${J.completed?'<svg class="w-4 h-4 text-green-500" fill="currentColor" viewBox="0 0 20 20"><path fill-rule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zm3.707-9.293a1 1 0 00-1.414-1.414L9 10.586 7.707 9.293a1 1 0 00-1.414 1.414l2 2a1 1 0 001.414 0l4-4z" clip-rule="evenodd"/></svg>':'<svg class="w-4 h-4 text-stone-400" fill="none" stroke="currentColor" viewBox="0 0 24 24"><circle cx="12" cy="12" r="10" stroke-width="2"/></svg>'}
                            <span class="${J.completed?"text-stone-700 dark:text-stone-300":"text-stone-500 dark:text-stone-400"}">${J.label}</span>
                        </div>
                    `).join("")}
                </div>
            </div>
        `}async function Y(){if(AuthApi.isAuthenticated())try{n=(await AuthApi.getProfile()).data,z()}catch{Toast.error("Failed to load profile.")}}function Z(){let i=document.querySelectorAll("[data-profile-tab]"),a=document.querySelectorAll("[data-profile-panel]");if(!i.length||!a.length)return;let c=m=>{a.forEach(b=>{b.classList.toggle("hidden",b.dataset.profilePanel!==m)}),i.forEach(b=>{let v=b.dataset.profileTab===m;b.classList.toggle("bg-amber-600",v),b.classList.toggle("text-white",v),b.classList.toggle("shadow-sm",v),b.classList.toggle("text-stone-700",!v),b.classList.toggle("dark:text-stone-200",!v)}),localStorage.setItem("profileTab",m)},u=localStorage.getItem("profileTab")||"overview";c(u),i.forEach(m=>{m.addEventListener("click",()=>{c(m.dataset.profileTab)})})}function z(){let i=document.getElementById("profile-info");if(!i||!n)return;let a=`${Templates.escapeHtml(n.first_name||"")} ${Templates.escapeHtml(n.last_name||"")}`.trim()||Templates.escapeHtml(n.email||"User"),c=Templates.formatDate(n.created_at||n.date_joined),u=n.avatar?`<img id="avatar-preview" src="${n.avatar}" alt="Profile" class="w-full h-full object-cover">`:`
            <span class="flex h-full w-full items-center justify-center text-3xl font-semibold text-stone-500">
                ${(n.first_name?.[0]||n.email?.[0]||"U").toUpperCase()}
            </span>`;i.innerHTML=`
            <div class="absolute inset-0 bg-gradient-to-r from-amber-50/80 via-amber-100/60 to-transparent dark:from-amber-900/20 dark:via-amber-800/10" aria-hidden="true"></div>
            <div class="relative flex flex-col gap-4 md:flex-row md:items-center md:gap-6">
                <div class="relative">
                    <div class="w-24 h-24 rounded-2xl ring-4 ring-amber-100 dark:ring-amber-900/40 overflow-hidden bg-stone-100 dark:bg-stone-800">
                        ${u}
                    </div>
                </div>
                <div class="flex-1 min-w-0">
                    <p class="text-sm font-semibold text-amber-700 dark:text-amber-300">Profile</p>
                    <h1 class="text-2xl font-bold text-stone-900 dark:text-white leading-tight truncate">${a}</h1>
                    <p class="text-sm text-stone-600 dark:text-stone-300 truncate">${Templates.escapeHtml(n.email)}</p>
                    <p class="text-xs text-stone-500 dark:text-stone-400 mt-1">Member since ${c}</p>
                    <div class="flex flex-wrap gap-2 mt-4">
                        <button type="button" id="change-avatar-btn" class="btn btn-primary btn-sm">Update photo</button>
                        ${n.avatar?'<button type="button" id="remove-avatar-btn" class="btn btn-ghost btn-sm text-red-600 hover:text-red-700 dark:text-red-400">Remove photo</button>':""}
                    </div>
                    <p class="text-xs text-stone-500 dark:text-stone-400 mt-3">JPG, GIF or PNG. Max size 5MB.</p>
                </div>
            </div>
        `,U()}function E(){Tabs.init()}function H(){let i=document.getElementById("profile-form");if(!i||!n)return;let a=document.getElementById("profile-first-name"),c=document.getElementById("profile-last-name"),u=document.getElementById("profile-email"),m=document.getElementById("profile-phone");a&&(a.value=n.first_name||""),c&&(c.value=n.last_name||""),u&&(u.value=n.email||""),m&&(m.value=n.phone||""),i.addEventListener("submit",async b=>{b.preventDefault();let v=new FormData(i),P={first_name:v.get("first_name"),last_name:v.get("last_name"),phone:v.get("phone")},D=i.querySelector('button[type="submit"]');D.disabled=!0,D.textContent="Saving...";try{await AuthApi.updateProfile(P),Toast.success("Profile updated successfully!"),await Y()}catch(J){Toast.error(J.message||"Failed to update profile.")}finally{D.disabled=!1,D.textContent="Save Changes"}})}function U(){let i=document.getElementById("avatar-input"),a=document.getElementById("change-avatar-btn"),c=document.getElementById("remove-avatar-btn");i||(i=document.createElement("input"),i.type="file",i.id="avatar-input",i.name="avatar",i.accept="image/*",i.className="hidden",document.body.appendChild(i)),document.querySelectorAll("#change-avatar-btn").forEach(b=>b.addEventListener("click",()=>i.click())),document.querySelectorAll("#remove-avatar-btn").forEach(b=>b.addEventListener("click",()=>{typeof window.removeAvatar=="function"&&window.removeAvatar()})),i.removeEventListener?.("change",window._avatarChangeHandler),window._avatarChangeHandler=async function(b){let v=b.target.files?.[0];if(v){if(!v.type.startsWith("image/")){Toast.error("Please select an image file.");return}if(v.size>5242880){Toast.error("Image must be smaller than 5MB.");return}try{await AuthApi.uploadAvatar(v),Toast.success("Avatar updated!"),await Y()}catch(P){Toast.error(P.message||"Failed to update avatar.")}}},i.addEventListener("change",window._avatarChangeHandler)}function B(){let i=document.getElementById("password-form");i&&i.addEventListener("submit",async a=>{a.preventDefault();let c=new FormData(i),u=c.get("current_password"),m=c.get("new_password"),b=c.get("confirm_password");if(m!==b){Toast.error("Passwords do not match.");return}if(m.length<8){Toast.error("Password must be at least 8 characters.");return}let v=i.querySelector('button[type="submit"]');v.disabled=!0,v.textContent="Updating...";try{await AuthApi.changePassword(u,m),Toast.success("Password updated successfully!"),i.reset()}catch(P){Toast.error(P.message||"Failed to update password.")}finally{v.disabled=!1,v.textContent="Update Password"}})}function C(){k(),document.getElementById("add-address-btn")?.addEventListener("click",()=>{T()})}async function k(){let i=document.getElementById("addresses-list");if(i){Loader.show(i,"spinner");try{let c=(await AuthApi.getAddresses()).data||[];if(c.length===0){i.innerHTML=`
                    <div class="text-center py-8">
                        <svg class="w-16 h-16 text-gray-300 mx-auto mb-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M17.657 16.657L13.414 20.9a1.998 1.998 0 01-2.827 0l-4.244-4.243a8 8 0 1111.314 0z"/>
                            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M15 11a3 3 0 11-6 0 3 3 0 016 0z"/>
                        </svg>
                        <p class="text-gray-500">No saved addresses yet.</p>
                    </div>
                `;return}i.innerHTML=`
                <div class="grid grid-cols-1 md:grid-cols-2 gap-4">
                    ${c.map(u=>`
                        <div class="p-4 border border-gray-200 rounded-lg relative" data-address-id="${u.id}">
                            ${u.is_default?`
                                <span class="absolute top-2 right-2 px-2 py-0.5 bg-primary-100 text-primary-700 text-xs rounded">Default</span>
                            `:""}
                            <p class="font-medium text-gray-900">${Templates.escapeHtml(u.full_name||`${u.first_name} ${u.last_name}`)}</p>
                            <p class="text-sm text-gray-600 mt-1">${Templates.escapeHtml(u.address_line_1)}</p>
                            ${u.address_line_2?`<p class="text-sm text-gray-600">${Templates.escapeHtml(u.address_line_2)}</p>`:""}
                            <p class="text-sm text-gray-600">${Templates.escapeHtml(u.city)}, ${Templates.escapeHtml(u.state||"")} ${Templates.escapeHtml(u.postal_code)}</p>
                            <p class="text-sm text-gray-600">${Templates.escapeHtml(u.country)}</p>
                            ${u.phone?`<p class="text-sm text-gray-600 mt-1">${Templates.escapeHtml(u.phone)}</p>`:""}
                            
                            <div class="mt-4 flex gap-2">
                                <button class="edit-address-btn text-sm text-primary-600 hover:text-primary-700" data-address-id="${u.id}">Edit</button>
                                ${u.is_default?"":`
                                    <button class="set-default-btn text-sm text-gray-600 hover:text-gray-700" data-address-id="${u.id}">Set as Default</button>
                                `}
                                <button class="delete-address-btn text-sm text-red-600 hover:text-red-700" data-address-id="${u.id}">Delete</button>
                            </div>
                        </div>
                    `).join("")}
                </div>
            `,S()}catch(a){console.error("Failed to load addresses:",a),i.innerHTML='<p class="text-red-500">Failed to load addresses.</p>'}}}function S(){document.querySelectorAll(".edit-address-btn").forEach(i=>{i.addEventListener("click",async()=>{let a=i.dataset.addressId;try{let c=await AuthApi.getAddress(a);T(c.data)}catch{Toast.error("Failed to load address.")}})}),document.querySelectorAll(".set-default-btn").forEach(i=>{i.addEventListener("click",async()=>{let a=i.dataset.addressId;try{await AuthApi.setDefaultAddress(a),Toast.success("Default address updated."),await k()}catch{Toast.error("Failed to update default address.")}})}),document.querySelectorAll(".delete-address-btn").forEach(i=>{i.addEventListener("click",async()=>{let a=i.dataset.addressId;if(await Modal.confirm({title:"Delete Address",message:"Are you sure you want to delete this address?",confirmText:"Delete",cancelText:"Cancel"}))try{await AuthApi.deleteAddress(a),Toast.success("Address deleted."),await k()}catch{Toast.error("Failed to delete address.")}})})}function T(i=null){let a=!!i;Modal.open({title:a?"Edit Address":"Add New Address",content:`
                <form id="address-modal-form" class="space-y-4">
                    <div class="grid grid-cols-2 gap-4">
                        <div>
                            <label class="block text-sm font-medium text-gray-700 mb-1">First Name *</label>
                            <input type="text" name="first_name" value="${i?.first_name||""}" required class="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-primary-500 focus:border-primary-500">
                        </div>
                        <div>
                            <label class="block text-sm font-medium text-gray-700 mb-1">Last Name *</label>
                            <input type="text" name="last_name" value="${i?.last_name||""}" required class="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-primary-500 focus:border-primary-500">
                        </div>
                    </div>
                    <div>
                        <label class="block text-sm font-medium text-gray-700 mb-1">Phone</label>
                        <input type="tel" name="phone" value="${i?.phone||""}" class="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-primary-500 focus:border-primary-500">
                    </div>
                    <div>
                        <label class="block text-sm font-medium text-gray-700 mb-1">Address Line 1 *</label>
                        <input type="text" name="address_line_1" value="${i?.address_line_1||""}" required class="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-primary-500 focus:border-primary-500">
                    </div>
                    <div>
                        <label class="block text-sm font-medium text-gray-700 mb-1">Address Line 2</label>
                        <input type="text" name="address_line_2" value="${i?.address_line_2||""}" class="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-primary-500 focus:border-primary-500">
                    </div>
                    <div class="grid grid-cols-2 gap-4">
                        <div>
                            <label class="block text-sm font-medium text-gray-700 mb-1">City *</label>
                            <input type="text" name="city" value="${i?.city||""}" required class="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-primary-500 focus:border-primary-500">
                        </div>
                        <div>
                            <label class="block text-sm font-medium text-gray-700 mb-1">State/Province</label>
                            <input type="text" name="state" value="${i?.state||""}" class="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-primary-500 focus:border-primary-500">
                        </div>
                    </div>
                    <div class="grid grid-cols-2 gap-4">
                        <div>
                            <label class="block text-sm font-medium text-gray-700 mb-1">Postal Code *</label>
                            <input type="text" name="postal_code" value="${i?.postal_code||""}" required class="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-primary-500 focus:border-primary-500">
                        </div>
                        <div>
                            <label class="block text-sm font-medium text-gray-700 mb-1">Country *</label>
                            <select name="country" required class="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-primary-500 focus:border-primary-500">
                                <option value="">Select country</option>
                                <option value="BD" ${i?.country==="BD"?"selected":""}>Bangladesh</option>
                                <option value="US" ${i?.country==="US"?"selected":""}>United States</option>
                                <option value="UK" ${i?.country==="UK"?"selected":""}>United Kingdom</option>
                                <option value="CA" ${i?.country==="CA"?"selected":""}>Canada</option>
                                <option value="AU" ${i?.country==="AU"?"selected":""}>Australia</option>
                            </select>
                        </div>
                    </div>
                    <div>
                        <label class="flex items-center">
                            <input type="checkbox" name="is_default" ${i?.is_default?"checked":""} class="text-primary-600 focus:ring-primary-500 rounded">
                            <span class="ml-2 text-sm text-gray-600">Set as default address</span>
                        </label>
                    </div>
                </form>
            `,confirmText:a?"Save Changes":"Add Address",onConfirm:async()=>{let c=document.getElementById("address-modal-form"),u=new FormData(c),m={first_name:u.get("first_name"),last_name:u.get("last_name"),phone:u.get("phone"),address_line_1:u.get("address_line_1"),address_line_2:u.get("address_line_2"),city:u.get("city"),state:u.get("state"),postal_code:u.get("postal_code"),country:u.get("country"),is_default:u.get("is_default")==="on"};try{return a?(await AuthApi.updateAddress(i.id,m),Toast.success("Address updated!")):(await AuthApi.addAddress(m),Toast.success("Address added!")),await k(),!0}catch(b){return Toast.error(b.message||"Failed to save address."),!1}}})}function y(){n=null}return{init:e,destroy:y}})();window.AccountPage=Vs;jr=Vs});var Ys={};te(Ys,{default:()=>Nr});var Ws,Nr,Gs=ee(()=>{Ws=(function(){"use strict";let n=null,e=[],s=null,o=null,h=6e4,w="data-bound";function _(){}function L(g){if(g==null||g==="")return 0;if(typeof g=="number"&&Number.isFinite(g))return g;let $=parseFloat(g);return Number.isFinite($)?$:0}function A(g){let $=L(g);return Templates.formatPrice($)}function Y(){let g=window.BUNORAA_CART?.gift||{},$=g.gift_wrap_enabled!==!1,N=L(g.gift_wrap_amount??g.gift_wrap_cost??0),M=!!g.is_gift,q=M&&$&&!!g.gift_wrap,R=q?L(g.gift_wrap_cost??N):0;return{is_gift:M,gift_message:g.gift_message||"",gift_wrap:q,gift_wrap_cost:R,gift_wrap_amount:N,gift_wrap_label:g.gift_wrap_label||"Gift Wrap",gift_wrap_enabled:$}}function Z(g={}){o={...o||Y(),...g},o.is_gift||(o.gift_message="",o.gift_wrap=!1,o.gift_wrap_cost=0),o.gift_wrap_enabled||(o.gift_wrap=!1,o.gift_wrap_cost=0),o.gift_wrap&&o.gift_wrap_cost<=0&&(o.gift_wrap_cost=L(o.gift_wrap_amount))}function z(g){return g!=null&&g!==""}function E(){let g=window.BUNORAA_CART?.taxRate;return L(z(g)?g:0)}function H(){let g=window.BUNORAA_CART?.freeShippingThreshold,N=(window.BUNORAA_SHIPPING||{}).free_shipping_threshold??window.BUNORAA_CURRENCY?.free_shipping_threshold??0;return L(z(g)?g:N)}let U='<a href="/account/addresses/" class="text-xs font-medium text-amber-600 dark:text-amber-400 underline underline-offset-2 hover:text-amber-700 dark:hover:text-amber-300">Add shipping address to see shipping cost.</a>',B=null,C=null,k={key:null,quote:null};function S(){return(document.getElementById("delivery-division")?.value||"dhaka").toLowerCase()}function T(g){return g?g.charAt(0).toUpperCase()+g.slice(1):"Dhaka"}function y(){return window.AuthApi?.isAuthenticated&&AuthApi.isAuthenticated()}function i(g){return g?Array.isArray(g)?g:Array.isArray(g.data)?g.data:Array.isArray(g.data?.results)?g.data.results:Array.isArray(g.results)?g.results:[]:[]}async function a(){return y()?B||(C||(C=(async()=>{try{let g=await AuthApi.getAddresses(),$=i(g);if(!$.length)return null;let N=$.filter(M=>{let q=String(M.address_type||"").toLowerCase();return q==="shipping"||q==="both"});return N.find(M=>M.is_default)||N[0]||$.find(M=>M.is_default)||$[0]||null}catch{return null}})()),B=await C,B):null}function c(g){let $=E();return $>0?g*$/100:0}function u(g){let $=document.getElementById("shipping-location");if(!$)return;if(g&&(g.city||g.state||g.country)){$.textContent=g.city||g.state||g.country;return}let N=document.getElementById("delivery-division");$.textContent=N?T(S()):"Address"}function m(g,$,N,M){return{country:g?.country||"BD",state:g?.state||g?.city||"",postal_code:g?.postal_code||"",subtotal:$,weight:0,item_count:N,product_ids:M}}async function b(g,$,N){if(!N)return null;let M=g?.items||[],q=M.reduce((W,ne)=>W+Number(ne.quantity||1),0),R=M.map(W=>W.product?.id||W.product_id).filter(Boolean),X=m(N,$,q,R),G=[N.id||"",X.country,X.state,X.postal_code,$,q,R.join(",")].join("|");if(k.key===G)return k.quote;try{let W;if(window.ShippingApi?.calculateShipping)W=await ShippingApi.calculateShipping(X);else{let ve=window.BUNORAA_CURRENCY?.code;W=await fetch("/api/v1/shipping/calculate/",{method:"POST",headers:{"Content-Type":"application/json","X-CSRFToken":window.CSRF_TOKEN||document.querySelector("[name=csrfmiddlewaretoken]")?.value||"",...ve?{"X-User-Currency":ve}:{}},body:JSON.stringify(X)}).then(he=>he.json())}let ne=W?.data?.methods||W?.data?.data?.methods||W?.methods||[];if(!Array.isArray(ne)||ne.length===0)return null;let re=ne.reduce((ve,he)=>{let Le=L(ve.rate);return L(he.rate)<Le?he:ve},ne[0]),se=L(re.rate),ue={cost:se,isFree:re.is_free||se<=0,display:re.rate_display||A(se)};return k={key:G,quote:ue},ue}catch{return null}}async function v(g,$,N){let M=document.getElementById("shipping"),q=document.getElementById("tax"),R=document.getElementById("total"),X=document.getElementById("gift-wrap-row"),G=document.getElementById("gift-wrap-cost"),W=Math.max(0,g-$),ne=c(W),re=o?.gift_wrap?L(o.gift_wrap_cost):0;q&&(q.textContent=A(ne)),X&&G&&(o?.gift_wrap?(X.classList.remove("hidden"),G.textContent=`+${A(re)}`,G.dataset.price=re):(X.classList.add("hidden"),G.textContent=`+${A(0)}`,G.dataset.price=0));let se=await a();if(!se){M&&(M.innerHTML=U),R&&(R.textContent=A(W+ne+re)),u(null);return}M&&(M.textContent="Calculating...");let ue=await b(N||n,g,se);if(!ue){M&&(M.innerHTML=U),R&&(R.textContent=A(W+ne+re)),u(se);return}M&&(M.textContent=ue.isFree?"Free":ue.display),R&&(R.textContent=A(W+ne+re+ue.cost)),u(se)}async function P(){o=Y(),Z(o),await l(),ke(),Te(),fe()}function D(g,$,N){g&&(g.dataset.originalText||(g.dataset.originalText=g.textContent),g.disabled=$,g.textContent=$?N:g.dataset.originalText)}function J(g,$){let N=["request failed.","request failed","invalid response format","invalid request format"],M=G=>{if(!G)return!0;let W=String(G).trim().toLowerCase();return N.includes(W)},q=G=>{if(!G)return null;if(typeof G=="string")return G;if(Array.isArray(G))return G[0];if(typeof G=="object"){let W=Object.values(G),re=(W.flat?W.flat():W.reduce((se,ue)=>se.concat(ue),[]))[0]??W[0];if(typeof re=="string")return re;if(re&&typeof re=="object")return q(re)}return null},R=[];return g?.message&&R.push(g.message),g?.data?.message&&R.push(g.data.message),g?.data?.detail&&R.push(g.data.detail),g?.data&&typeof g.data=="string"&&R.push(g.data),g?.errors&&R.push(q(g.errors)),g?.data&&typeof g.data=="object"&&R.push(q(g.data)),R.find(G=>G&&!M(G))||$}function le(g){let $=document.getElementById("applied-coupon"),N=document.getElementById("coupon-name"),M=document.getElementById("coupon-form"),q=document.getElementById("coupon-code"),R=document.getElementById("apply-coupon-btn"),X=q?.closest("div.flex"),G=document.getElementById("coupon-message");if(g){$&&$.classList.remove("hidden"),N&&(N.textContent=g),M&&M.classList.add("hidden"),!M&&X&&X.classList.add("hidden"),R&&R.classList.add("hidden"),q&&(q.value=g),G&&G.classList.add("hidden");return}$&&$.classList.add("hidden"),M&&M.classList.remove("hidden"),!M&&X&&X.classList.remove("hidden"),R&&R.classList.remove("hidden"),q&&(q.value="")}function be(g){let $=document.getElementById("validation-messages"),N=document.getElementById("validation-issues"),M=document.getElementById("validation-warnings"),q=document.getElementById("issues-list"),R=document.getElementById("warnings-list");if(!$||!N||!M||!q||!R)return;let X=Array.isArray(g?.issues)?g.issues:[],G=Array.isArray(g?.warnings)?g.warnings:[];if(q.innerHTML=X.map(W=>`<li class="flex items-start gap-2"><span class="mt-1 w-1.5 h-1.5 rounded-full bg-red-500 flex-shrink-0"></span><span>${Templates.escapeHtml(W?.message||"Issue found")}</span></li>`).join(""),R.innerHTML=G.map(W=>`<li class="flex items-start gap-2"><span class="mt-1 w-1.5 h-1.5 rounded-full bg-amber-500 flex-shrink-0"></span><span>${Templates.escapeHtml(W?.message||"Warning")}</span></li>`).join(""),!X.length&&!G.length){$.classList.add("hidden"),N.classList.add("hidden"),M.classList.add("hidden");return}$.classList.remove("hidden"),N.classList.toggle("hidden",X.length===0),M.classList.toggle("hidden",G.length===0),$.scrollIntoView({behavior:"smooth",block:"start"})}function ke(){let g=document.getElementById("validate-cart-btn"),$=document.getElementById("lock-prices-btn"),N=document.getElementById("share-cart-btn");g&&g.getAttribute(w)!=="true"&&(g.setAttribute(w,"true"),g.addEventListener("click",async()=>{D(g,!0,"Validating...");try{let M=await CartApi.validateCart();be(M.data),(M.data?.issues||[]).length>0?Toast.warning("We found a few issues in your cart."):Toast.success("Cart looks good!")}catch(M){Toast.error(J(M,"Unable to validate cart right now."))}finally{D(g,!1)}})),$&&$.getAttribute(w)!=="true"&&($.setAttribute(w,"true"),$.addEventListener("click",async()=>{D($,!0,"Locking...");try{let q=(await CartApi.lockPrices()).data?.locked_count??0;Toast.success(`Locked prices for ${q} item${q===1?"":"s"}.`),await l()}catch(M){Toast.error(J(M,"Failed to lock prices."))}finally{D($,!1)}})),N&&N.getAttribute(w)!=="true"&&(N.setAttribute(w,"true"),N.addEventListener("click",async()=>{D(N,!0,"Creating...");try{let q=(await CartApi.shareCart({permission:"view",expires_days:7})).data?.share_url;if(!q)throw new Error("Share link unavailable.");navigator.share?(await navigator.share({title:"Shared Cart",url:q}),Toast.success("Share link ready.")):navigator.clipboard?.writeText?(await navigator.clipboard.writeText(q),Toast.success("Share link copied to clipboard.")):window.prompt("Copy this link to share your cart:",q)}catch(M){Toast.error(J(M,"Failed to create share link."))}finally{D(N,!1)}}))}function fe(){we(),ce(),me(),x(),t()}function we(){e=JSON.parse(localStorage.getItem("savedForLater")||"[]"),ye()}function ye(){let g=document.getElementById("saved-for-later");if(g){if(e.length===0){g.innerHTML="",g.classList.add("hidden");return}g.classList.remove("hidden"),g.innerHTML=`
            <div class="mt-8 bg-white dark:bg-stone-800 rounded-xl shadow-sm border border-gray-100 dark:border-stone-700 overflow-hidden">
                <div class="px-6 py-4 border-b border-gray-100 dark:border-stone-700 flex items-center justify-between">
                    <h3 class="text-lg font-semibold text-gray-900 dark:text-white">Saved for Later (${e.length})</h3>
                    <button id="clear-saved-btn" class="text-sm text-gray-500 dark:text-stone-400 hover:text-gray-700 dark:hover:text-stone-300">Clear All</button>
                </div>
                <div class="p-6 grid grid-cols-2 md:grid-cols-4 gap-4">
                    ${e.map($=>`
                        <div class="saved-item" data-product-id="${$.id}">
                            <div class="aspect-square bg-gray-100 dark:bg-stone-700 rounded-lg overflow-hidden mb-2">
                                <img src="${$.image||"/static/images/placeholder.jpg"}" alt="${Templates.escapeHtml($.name)}" class="w-full h-full object-cover">
                            </div>
                            <h4 class="text-sm font-medium text-gray-900 dark:text-white truncate">${Templates.escapeHtml($.name)}</h4>
                            <p class="text-sm font-semibold text-primary-600 dark:text-amber-400">${A($.price)}</p>
                            <div class="flex gap-2 mt-2">
                                <button class="move-to-cart-btn flex-1 px-2 py-1 bg-primary-600 dark:bg-amber-600 text-white text-xs font-medium rounded hover:bg-primary-700 dark:hover:bg-amber-700 transition-colors">Move to Cart</button>
                                <button class="remove-saved-btn px-2 py-1 text-gray-400 hover:text-red-500">
                                    <svg class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M6 18L18 6M6 6l12 12"/></svg>
                                </button>
                            </div>
                        </div>
                    `).join("")}
                </div>
            </div>
        `,g.querySelectorAll(".move-to-cart-btn").forEach($=>{$.addEventListener("click",async()=>{let M=$.closest(".saved-item")?.dataset.productId;if(M)try{await CartApi.addItem(M,1),e=e.filter(q=>q.id!==M),localStorage.setItem("savedForLater",JSON.stringify(e)),Toast.success("Item moved to cart"),await l(),ye()}catch{Toast.error("Failed to move item to cart")}})}),g.querySelectorAll(".remove-saved-btn").forEach($=>{$.addEventListener("click",()=>{let M=$.closest(".saved-item")?.dataset.productId;M&&(e=e.filter(q=>q.id!==M),localStorage.setItem("savedForLater",JSON.stringify(e)),ye(),Toast.info("Item removed"))})}),document.getElementById("clear-saved-btn")?.addEventListener("click",()=>{e=[],localStorage.removeItem("savedForLater"),ye(),Toast.info("Saved items cleared")})}}function ce(){let g=JSON.parse(localStorage.getItem("abandonedCart")||"null");g&&g.items?.length>0&&(!n||n.items?.length===0)&&pe(g),xe(),window.addEventListener("beforeunload",()=>{n&&n.items?.length>0&&localStorage.setItem("abandonedCart",JSON.stringify({items:n.items,savedAt:new Date().toISOString()}))})}function xe(){let g=()=>{s&&clearTimeout(s),s=setTimeout(()=>{n&&n.items?.length>0&&Ee()},h)};["click","scroll","keypress","mousemove"].forEach($=>{document.addEventListener($,g,{passive:!0,once:!1})}),g()}function pe(g){let $=document.createElement("div");$.id="abandoned-cart-modal",$.className="fixed inset-0 z-50 flex items-center justify-center p-4",$.innerHTML=`
            <div class="absolute inset-0 bg-black/50 backdrop-blur-sm" onclick="document.getElementById('abandoned-cart-modal').remove()"></div>
            <div class="relative bg-white dark:bg-stone-800 rounded-2xl shadow-2xl max-w-md w-full p-6">
                <button onclick="document.getElementById('abandoned-cart-modal').remove()" class="absolute top-4 right-4 w-8 h-8 bg-stone-100 dark:bg-stone-700 rounded-full flex items-center justify-center">
                    <svg class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M6 18L18 6M6 6l12 12"/></svg>
                </button>
                <div class="text-center mb-6">
                    <div class="w-16 h-16 bg-amber-100 dark:bg-amber-900/30 rounded-full flex items-center justify-center mx-auto mb-4">
                        <svg class="w-8 h-8 text-amber-600 dark:text-amber-400" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M3 3h2l.4 2M7 13h10l4-8H5.4M7 13L5.4 5M7 13l-2.293 2.293c-.63.63-.184 1.707.707 1.707H17m0 0a2 2 0 100 4 2 2 0 000-4zm-8 2a2 2 0 11-4 0 2 2 0 014 0z"/></svg>
                    </div>
                    <h3 class="text-xl font-bold text-stone-900 dark:text-white mb-2">Welcome Back!</h3>
                    <p class="text-stone-600 dark:text-stone-400">You left ${g.items.length} item(s) in your cart.</p>
                </div>
                <div class="max-h-48 overflow-y-auto mb-6 space-y-2">
                    ${g.items.slice(0,3).map(N=>`
                        <div class="flex items-center gap-3 p-2 bg-stone-50 dark:bg-stone-700/50 rounded-lg">
                            <img src="${N.product_image||N.product?.image||"/static/images/placeholder.jpg"}" alt="" class="w-12 h-12 rounded object-cover">
                            <div class="flex-1 min-w-0">
                                <p class="text-sm font-medium text-stone-900 dark:text-white truncate">${Templates.escapeHtml(N.product_name||N.product?.name||"Product")}</p>
                                <p class="text-xs text-stone-500 dark:text-stone-400">Qty: ${N.quantity}</p>
                            </div>
                        </div>
                    `).join("")}
                    ${g.items.length>3?`<p class="text-center text-sm text-stone-500 dark:text-stone-400">+${g.items.length-3} more items</p>`:""}
                </div>
                <div class="grid grid-cols-2 gap-3">
                    <button onclick="document.getElementById('abandoned-cart-modal').remove(); localStorage.removeItem('abandonedCart');" class="py-3 px-4 border border-stone-300 dark:border-stone-600 text-stone-700 dark:text-stone-300 font-medium rounded-xl hover:bg-stone-50 dark:hover:bg-stone-700 transition-colors">
                        Start Fresh
                    </button>
                    <button onclick="document.getElementById('abandoned-cart-modal').remove();" class="py-3 px-4 bg-primary-600 dark:bg-amber-600 text-white font-semibold rounded-xl hover:bg-primary-700 dark:hover:bg-amber-700 transition-colors">
                        Continue Shopping
                    </button>
                </div>
            </div>
        `,document.body.appendChild($)}function Ee(){sessionStorage.getItem("cartReminderShown")||(Toast.info("Don't forget! You have items in your cart",{duration:8e3,action:{text:"Checkout",onClick:()=>window.location.href="/checkout/"}}),sessionStorage.setItem("cartReminderShown","true"))}function me(){Ce()}function Ce(){let g=document.getElementById("free-shipping-progress");if(!g||!n)return;let $=L(window.BUNORAA_CART?.freeShippingThreshold??50),N=L(n.summary?.subtotal||n.subtotal||0),M=Math.max(0,$-N),q=Math.min(100,N/$*100);if(g.dataset.freeShippingProgress==="bar"){let R=document.getElementById("free-shipping-status"),X=document.getElementById("free-shipping-message");g.style.width=`${q}%`,N>=$?(R&&(R.textContent="Unlocked"),X&&(X.textContent="You've unlocked free delivery")):(R&&(R.textContent=`${A(M)} away`),X&&(X.textContent=`Free delivery on orders over ${A($)}`));return}N>=$?g.innerHTML=`
                <div class="flex items-center gap-2 p-3 bg-emerald-50 dark:bg-emerald-900/20 rounded-xl">
                    <svg class="w-5 h-5 text-emerald-600 dark:text-emerald-400" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M5 13l4 4L19 7"/></svg>
                    <span class="text-sm font-medium text-emerald-700 dark:text-emerald-300">You've unlocked FREE shipping!</span>
                </div>
            `:g.innerHTML=`
                <div class="p-3 bg-amber-50 dark:bg-amber-900/20 rounded-xl">
                    <div class="flex items-center justify-between text-sm mb-2">
                        <span class="text-amber-700 dark:text-amber-300">Add ${A(M)} for FREE shipping</span>
                        <span class="text-amber-600 dark:text-amber-400 font-medium">${Math.round(q)}%</span>
                    </div>
                    <div class="w-full bg-amber-200 dark:bg-amber-800 rounded-full h-2">
                        <div class="bg-amber-500 h-2 rounded-full transition-all duration-500" style="width: ${q}%"></div>
                    </div>
                </div>
            `}function _e(){let g=document.getElementById("cart-delivery-estimate");if(!g)return;let $=new Date,N=3,M=7,q=new Date($.getTime()+N*24*60*60*1e3),R=new Date($.getTime()+M*24*60*60*1e3),X=G=>G.toLocaleDateString("en-US",{weekday:"short",month:"short",day:"numeric"});g.innerHTML=`
            <div class="flex items-center gap-2 text-sm text-stone-600 dark:text-stone-400">
                <svg class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M5 8h14M5 8a2 2 0 110-4h14a2 2 0 110 4M5 8v10a2 2 0 002 2h10a2 2 0 002-2V8m-9 4h4"/></svg>
                <span>Estimated delivery: <strong class="text-stone-900 dark:text-white">${X(q)} - ${X(R)}</strong></span>
            </div>
        `}function $e(){let g=document.getElementById("delivery-division");if(!g)return;let $=()=>{let N=n?.summary||{},M=L(N.subtotal??n?.subtotal??0),q=L(N.discount_amount??n?.discount_amount??0);v(M,q,n)};g.addEventListener("change",$),$()}async function x(){let g=document.getElementById("cart-recommendations");if(!(!g||!n||!n.items?.length))try{let $=n.items.map(q=>q.product?.id||q.product_id).filter(Boolean);if(!$.length)return;let N=await ProductsApi.getRelated($[0],{limit:4}),M=N?.data||N?.results||[];if(!M.length)return;g.innerHTML=`
                <div class="mt-8">
                    <h3 class="text-lg font-semibold text-gray-900 dark:text-white mb-4">You might also like</h3>
                    <div class="grid grid-cols-2 md:grid-cols-4 gap-4">
                        ${M.slice(0,4).map(q=>`
                            <div class="bg-white dark:bg-stone-800 rounded-xl shadow-sm border border-gray-100 dark:border-stone-700 overflow-hidden group">
                                <a href="/products/${q.slug}/" class="block aspect-square bg-gray-100 dark:bg-stone-700 overflow-hidden">
                                    <img src="${q.primary_image||q.image||"/static/images/placeholder.jpg"}" alt="${Templates.escapeHtml(q.name)}" class="w-full h-full object-cover group-hover:scale-105 transition-transform duration-300">
                                </a>
                                <div class="p-3">
                                    <h4 class="text-sm font-medium text-gray-900 dark:text-white truncate">${Templates.escapeHtml(q.name)}</h4>
                                    <p class="text-sm font-semibold text-primary-600 dark:text-amber-400 mt-1">${A(q.current_price||q.price)}</p>
                                    <button class="quick-add-btn w-full mt-2 py-2 text-xs font-medium text-primary-600 dark:text-amber-400 border border-primary-600 dark:border-amber-400 rounded-lg hover:bg-primary-50 dark:hover:bg-amber-900/20 transition-colors" data-product-id="${q.id}">
                                        + Add to Cart
                                    </button>
                                </div>
                            </div>
                        `).join("")}
                    </div>
                </div>
            `,g.querySelectorAll(".quick-add-btn").forEach(q=>{q.addEventListener("click",async()=>{let R=q.dataset.productId;q.disabled=!0,q.textContent="Adding...";try{await CartApi.addItem(R,1),Toast.success("Added to cart"),await l()}catch{Toast.error("Failed to add item")}finally{q.disabled=!1,q.textContent="+ Add to Cart"}})})}catch($){console.warn("Failed to load recommendations:",$)}}function t(){let g=document.getElementById("gift-order"),$=document.getElementById("gift-details"),N=document.getElementById("gift-message"),M=document.getElementById("gift-message-count"),q=document.getElementById("gift-wrap");if(!g||g.getAttribute(w)==="true")return;g.setAttribute(w,"true");let R=()=>{N&&M&&(M.textContent=String(N.value.length))},X=()=>{o||(o=Y()),g.checked=!!o.is_gift,$&&$.classList.toggle("hidden",!o.is_gift),N&&(N.value=o.gift_message||""),q&&(q.checked=!!o.gift_wrap,q.disabled=!o.gift_wrap_enabled||!o.is_gift),R()},G=r(async()=>{try{let W={is_gift:!!o?.is_gift,gift_message:o?.gift_message||"",gift_wrap:!!o?.gift_wrap},ne=await CartApi.updateGiftOptions(W);if(ne?.success===!1)throw new Error(ne.message||"Failed to update gift options.");let re=ne?.data||ne,se=re?.gift_state;se&&Z({is_gift:se.is_gift,gift_message:se.gift_message||"",gift_wrap:se.gift_wrap,gift_wrap_cost:L(se.gift_wrap_cost)}),re?.gift_wrap_amount!==void 0&&(o.gift_wrap_amount=L(re.gift_wrap_amount)),re?.gift_wrap_label&&(o.gift_wrap_label=re.gift_wrap_label),re?.gift_wrap_enabled!==void 0&&(o.gift_wrap_enabled=!!re.gift_wrap_enabled),X();let ue=n?.summary||{},ve=L(ue.subtotal??n?.subtotal??0),he=L(ue.discount_amount??n?.discount_amount??0);v(ve,he,n)}catch(W){Toast.error(J(W,"Failed to update gift options."))}},400);g.addEventListener("change",()=>{Z({is_gift:g.checked}),g.checked||Z({gift_message:"",gift_wrap:!1,gift_wrap_cost:0}),X();let W=n?.summary||{},ne=L(W.subtotal??n?.subtotal??0),re=L(W.discount_amount??n?.discount_amount??0);v(ne,re,n),G()}),N?.addEventListener("input",()=>{o?.is_gift&&(Z({gift_message:N.value.slice(0,200)}),R(),G())}),q?.addEventListener("change",()=>{if(!o?.is_gift||!o?.gift_wrap_enabled){q.checked=!1;return}Z({gift_wrap:q.checked});let W=n?.summary||{},ne=L(W.subtotal??n?.subtotal??0),re=L(W.discount_amount??n?.discount_amount??0);v(ne,re,n),G()}),X()}function r(g,$){let N;return function(...q){clearTimeout(N),N=setTimeout(()=>g(...q),$)}}function d(){let g=document.getElementById("express-checkout");g&&(g.innerHTML=`
            <div class="mt-4 space-y-2">
                <p class="text-xs text-center text-stone-500 dark:text-stone-400 mb-3">Or checkout faster with</p>
                <div class="grid grid-cols-2 gap-2">
                    <button class="express-pay-btn flex items-center justify-center gap-2 px-4 py-2.5 bg-black text-white rounded-lg font-medium text-sm hover:bg-gray-800 transition-colors">
                        <svg class="w-5 h-5" viewBox="0 0 24 24" fill="currentColor"><path d="M17.05 20.28c-.98.95-2.05.8-3.08.35-1.09-.46-2.09-.48-3.24 0-1.44.62-2.2.44-3.06-.35C2.79 15.25 3.51 7.59 9.05 7.31c1.35.07 2.29.74 3.08.8 1.18-.24 2.31-.93 3.57-.84 1.51.12 2.65.72 3.4 1.8-3.12 1.87-2.38 5.98.48 7.13-.57 1.5-1.31 2.99-2.53 4.08zM12.03 7.25c-.15-2.23 1.66-4.07 3.74-4.25.27 2.43-2.22 4.38-3.74 4.25z"/></svg>
                        Apple Pay
                    </button>
                    <button class="express-pay-btn flex items-center justify-center gap-2 px-4 py-2.5 bg-[#5f6368] text-white rounded-lg font-medium text-sm hover:bg-[#4a4e52] transition-colors">
                        <svg class="w-5 h-5" viewBox="0 0 24 24" fill="currentColor"><path d="M22.56 12.25c0-.78-.07-1.53-.2-2.25H12v4.26h5.92c-.26 1.37-1.04 2.53-2.21 3.31v2.77h3.57c2.08-1.92 3.28-4.74 3.28-8.09z"/><path d="M12 23c2.97 0 5.46-.98 7.28-2.66l-3.57-2.77c-.98.66-2.23 1.06-3.71 1.06-2.86 0-5.29-1.93-6.16-4.53H2.18v2.84C3.99 20.53 7.7 23 12 23z"/><path d="M5.84 14.09c-.22-.66-.35-1.36-.35-2.09s.13-1.43.35-2.09V7.07H2.18C1.43 8.55 1 10.22 1 12s.43 3.45 1.18 4.93l2.85-2.22.81-.62z"/><path d="M12 5.38c1.62 0 3.06.56 4.21 1.64l3.15-3.15C17.45 2.09 14.97 1 12 1 7.7 1 3.99 3.47 2.18 7.07l3.66 2.84c.87-2.6 3.3-4.53 6.16-4.53z"/></svg>
                        Google Pay
                    </button>
                </div>
            </div>
        `,g.querySelectorAll(".express-pay-btn").forEach($=>{$.addEventListener("click",()=>{Toast.info("Express checkout coming soon!")})}))}async function l(){let g=document.getElementById("cart-container");if(g){g.dataset.cartRender!=="server"&&Loader.show(g,"skeleton");try{n=(await CartApi.getCart()).data,g.dataset.cartRender==="server"?p(n):f(n)}catch($){console.error("Failed to load cart:",$),g.innerHTML='<p class="text-red-500 text-center py-8">Failed to load cart. Please try again.</p>'}}}function p(g){let $=document.getElementById("cart-container");if(!$)return;let N=g?.items||[],M=g?.summary||{};if(!N.length){$.innerHTML=`
                <div class="text-center py-16">
                    <svg class="w-24 h-24 text-gray-300 dark:text-gray-600 mx-auto mb-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path stroke-linecap="round" stroke-linejoin="round" stroke-width="1.5" d="M3 3h2l.4 2M7 13h10l4-8H5.4M7 13L5.4 5M7 13l-2.293 2.293c-.63.63-.184 1.707.707 1.707H17m0 0a2 2 0 100 4 2 2 0 000-4zm-8 2a2 2 0 11-4 0 2 2 0 014 0z"/>
                    </svg>
                    <h2 class="text-2xl font-semibold text-gray-900 dark:text-white mb-2">Your cart is empty</h2>
                    <p class="text-gray-600 dark:text-gray-300 mb-6">Explore the collection and add your favorites to the bag.</p>
                    <a href="/products/" class="inline-block bg-gradient-to-r from-amber-600 to-amber-700 text-white px-8 py-3 rounded-xl font-semibold hover:from-amber-700 hover:to-amber-800 transition-colors shadow-lg shadow-amber-500/20">
                        Start Shopping
                    </a>
                </div>
            `;return}let q=M.subtotal??g?.subtotal??0,R=L(M.discount_amount??g?.discount_amount),X=document.getElementById("subtotal"),G=document.getElementById("discount-row"),W=document.getElementById("discount"),ne=document.getElementById("savings-row"),re=document.getElementById("savings");if(X&&(X.textContent=A(q)),G&&W&&(R>0?(G.classList.remove("hidden"),W.textContent=`-${A(R)}`):G.classList.add("hidden")),ne&&re){let se=L(M.total_savings);se>0?(ne.classList.remove("hidden"),re.textContent=A(se)):ne.classList.add("hidden")}v(q,R,g),le(g?.coupon_code||g?.coupon?.code||M?.coupon_code||""),N.forEach(se=>{let ue=$.querySelector(`.cart-item[data-item-id="${se.id}"]`);if(!ue)return;let ve=ue.querySelector(".qty-input");ve&&se.quantity&&(ve.value=se.quantity);let he=ue.querySelector(".item-total");he&&(he.textContent=A(se.total||se.line_total||0));let Le=ue.querySelector(".item-unit-price");Le&&(Le.textContent=A(se.unit_price||se.current_price||se.price_at_add||0));let Ie=ue.querySelector("img");Ie&&se.product_image&&(Ie.src=se.product_image)}),F(),Ce()}function f(g){let $=document.getElementById("cart-container");if(!$)return;let N=g?.items||[],M=g?.summary||{},q=M.subtotal??g?.subtotal??0,R=L(M.discount_amount??g?.discount_amount),X=Math.max(0,q-R),G=c(X),W=o?.gift_wrap?L(o.gift_wrap_cost):0,ne=X+G+W,re=E(),se=g?.coupon?.code||g?.coupon_code||"";if(N.length===0){$.innerHTML=`
                <div class="text-center py-16">
                    <svg class="w-24 h-24 text-gray-300 mx-auto mb-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path stroke-linecap="round" stroke-linejoin="round" stroke-width="1.5" d="M3 3h2l.4 2M7 13h10l4-8H5.4M7 13L5.4 5M7 13l-2.293 2.293c-.63.63-.184 1.707.707 1.707H17m0 0a2 2 0 100 4 2 2 0 000-4zm-8 2a2 2 0 11-4 0 2 2 0 014 0z"/>
                    </svg>
                    <h2 class="text-2xl font-bold text-gray-900 mb-2">Your cart is empty</h2>
                    <p class="text-gray-600 mb-8">Looks like you haven't added any items to your cart yet.</p>
                    <a href="/products/" class="inline-flex items-center px-6 py-3 bg-primary-600 text-white font-semibold rounded-lg hover:bg-primary-700 transition-colors">
                        Start Shopping
                        <svg class="w-5 h-5 ml-2" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M17 8l4 4m0 0l-4 4m4-4H3"/>
                        </svg>
                    </a>
                </div>
            `;return}$.innerHTML=`
            <div class="grid grid-cols-1 lg:grid-cols-3 gap-8">
                <!-- Cart Items -->
                <div class="lg:col-span-2">
                    <div class="bg-white rounded-xl shadow-sm border border-gray-100 overflow-hidden">
                        <div class="px-6 py-4 border-b border-gray-100">
                            <h2 class="text-lg font-semibold text-gray-900">Shopping Cart (${N.length} items)</h2>
                        </div>
                        <div id="cart-items" class="divide-y divide-gray-100">
                            ${N.map(ue=>I(ue)).join("")}
                        </div>
                    </div>

                    <!-- Continue Shopping -->
                    <div class="mt-6 flex items-center justify-between">
                        <a href="/products/" class="inline-flex items-center text-primary-600 hover:text-primary-700 font-medium">
                            <svg class="w-5 h-5 mr-2" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M7 16l-4-4m0 0l4-4m-4 4h18"/>
                            </svg>
                            Continue Shopping
                        </a>
                        <button id="clear-cart-btn" class="text-red-600 hover:text-red-700 font-medium">
                            Clear Cart
                        </button>
                    </div>
                    
                    <!-- Saved for Later -->
                    <div id="saved-for-later"></div>
                    
                    <!-- Recommendations -->
                    <div id="cart-recommendations"></div>
                </div>

                <!-- Order Summary -->
                <div class="lg:col-span-1">
                    <div class="bg-white dark:bg-stone-800 rounded-xl shadow-sm border border-gray-100 dark:border-stone-700 p-6 sticky top-4">
                        <h3 class="text-lg font-semibold text-gray-900 dark:text-white mb-4">Order Summary</h3>
                        
                        <!-- Free Shipping Progress -->
                        <div id="free-shipping-progress" class="mb-4"></div>
                        
                        <div class="space-y-3 text-sm">
                            <div class="flex justify-between">
                                <span class="text-gray-600 dark:text-stone-400">Subtotal</span>
                                <span class="font-medium text-gray-900 dark:text-white">${A(q)}</span>
                            </div>
                            ${R>0?`
                                <div class="flex justify-between text-green-600 dark:text-green-400">
                                    <span>Discount</span>
                                    <span>-${A(R)}</span>
                                </div>
                            `:""}
                            <div class="flex justify-between">
                                <span class="text-gray-600 dark:text-stone-400">Shipping</span>
                                <span id="shipping" class="font-medium text-gray-900 dark:text-white">Calculating...</span>
                            </div>
                            <div id="gift-wrap-row" class="flex justify-between ${o?.gift_wrap?"":"hidden"}">
                                <span class="text-gray-600 dark:text-stone-400">${Templates.escapeHtml(o?.gift_wrap_label||"Gift Wrap")}</span>
                                <span id="gift-wrap-cost" class="font-medium text-gray-900 dark:text-white" data-price="${W}">+${A(W)}</span>
                            </div>
                            <div class="flex justify-between">
                                <span class="text-gray-600 dark:text-stone-400">VAT (${re}%)</span>
                                <span id="tax" class="font-medium text-gray-900 dark:text-white">${A(G)}</span>
                            </div>
                            <div class="pt-3 border-t border-gray-200 dark:border-stone-600">
                                <div class="flex justify-between">
                                    <span class="text-base font-semibold text-gray-900 dark:text-white">Total</span>
                                    <span id="total" class="text-base font-bold text-gray-900 dark:text-white">${A(ne)}</span>
                                </div>
                                <p class="text-xs text-gray-500 dark:text-stone-400 mt-1">Shipping calculated from your saved address</p>
                            </div>
                        </div>
                        
                        <!-- Coupon Form -->
                        <div class="mt-6 pt-6 border-t border-gray-200 dark:border-stone-600">
                            ${se?`
                                <div class="flex items-center justify-between px-3 py-2 bg-green-50 dark:bg-green-900/20 rounded-lg">
                                    <p class="text-sm font-medium text-green-700 dark:text-green-400">
                                        Coupon Applied: <span class="font-semibold">${Templates.escapeHtml(se)}</span>
                                    </p>
                                    <button id="remove-coupon-btn" class="text-green-600 dark:text-green-400 hover:text-green-700 dark:hover:text-green-300" aria-label="Remove coupon">
                                        <svg class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M6 18L18 6M6 6l12 12"/>
                                        </svg>
                                    </button>
                                </div>
                            `:`
                                <form id="coupon-form" class="flex gap-2">
                                    <input 
                                        type="text" 
                                        id="coupon-code" 
                                        placeholder="Enter coupon code"
                                        class="flex-1 px-3 py-2 border border-gray-300 dark:border-stone-600 dark:bg-stone-700 dark:text-white rounded-lg text-sm focus:ring-primary-500 dark:focus:ring-amber-500 focus:border-primary-500 dark:focus:border-amber-500"
                                        value="${Templates.escapeHtml(se)}"
                                    >
                                    <button type="submit" class="px-4 py-2 bg-gray-100 dark:bg-stone-700 hover:bg-gray-200 dark:hover:bg-stone-600 text-gray-700 dark:text-stone-200 font-medium rounded-lg transition-colors text-sm">
                                        Apply
                                    </button>
                                </form>
                            `}
                        </div>

                        <!-- Gift Options -->
                        <div class="mt-6 pt-6 border-t border-gray-200 dark:border-stone-600">
                            <h4 class="text-sm font-semibold text-gray-900 dark:text-white mb-3">Gift options</h4>
                            <label class="flex items-center gap-2 cursor-pointer">
                                <input type="checkbox" id="gift-order" class="rounded border-gray-300 dark:border-stone-600 text-primary-600 dark:text-amber-500 focus:ring-primary-500 dark:focus:ring-amber-500" ${o?.is_gift?"checked":""}>
                                <span class="text-sm text-gray-700 dark:text-stone-300">This order is a gift</span>
                            </label>
                            <div id="gift-details" class="${o?.is_gift?"":"hidden"} mt-3 space-y-3 pl-6">
                                <div>
                                    <label class="block text-xs font-medium text-gray-600 dark:text-stone-400 mb-1">Gift message (optional)</label>
                                    <textarea id="gift-message" maxlength="200" rows="3"
                                              class="w-full px-3 py-2 border border-gray-300 dark:border-stone-600 dark:bg-stone-700 dark:text-white rounded-lg text-sm focus:ring-primary-500 dark:focus:ring-amber-500 focus:border-primary-500 dark:focus:border-amber-500"
                                              placeholder="Add a personal message (max 200 characters)">${Templates.escapeHtml(o?.gift_message||"")}</textarea>
                                    <p class="text-xs text-gray-500 dark:text-stone-400 mt-1"><span id="gift-message-count">0</span>/200 characters</p>
                                </div>
                                <label class="flex items-center gap-2 cursor-pointer">
                                    <input type="checkbox" id="gift-wrap"
                                           class="rounded border-gray-300 dark:border-stone-600 text-primary-600 dark:text-amber-500 focus:ring-primary-500 dark:focus:ring-amber-500"
                                           ${o?.gift_wrap?"checked":""} ${o?.gift_wrap_enabled?"":"disabled"}>
                                    <span class="text-sm text-gray-700 dark:text-stone-300">Add ${Templates.escapeHtml(o?.gift_wrap_label||"Gift Wrap")}${o?.gift_wrap_enabled?` (+${A(o?.gift_wrap_amount||W)})`:" (Unavailable)"}</span>
                                </label>
                            </div>
                        </div>

                        <!-- Checkout Button -->
                        <a href="${window.BUNORAA_CART&&window.BUNORAA_CART.checkoutUrl?window.BUNORAA_CART.checkoutUrl:"/checkout/"}" class="mt-6 w-full px-6 py-3 bg-primary-600 dark:bg-amber-600 text-white font-semibold rounded-lg hover:bg-primary-700 dark:hover:bg-amber-700 transition-colors flex items-center justify-center gap-2">
                            Proceed to Checkout
                            <svg class="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M17 8l4 4m0 0l-4 4m4-4H3"/>
                            </svg>
                        </a>
                        
                        <!-- Trust Badges -->
                        <div class="mt-6 pt-6 border-t border-gray-200 dark:border-stone-600">
                            <div class="flex items-center justify-center gap-4 text-gray-400 dark:text-stone-500">
                                <div class="flex flex-col items-center">
                                    <svg class="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                        <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 15v2m-6 4h12a2 2 0 002-2v-6a2 2 0 00-2-2H6a2 2 0 00-2 2v6a2 2 0 002 2zm10-10V7a4 4 0 00-8 0v4h8z"/>
                                    </svg>
                                    <span class="text-xs mt-1">Secure</span>
                                </div>
                                <div class="flex flex-col items-center">
                                    <svg class="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                        <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9 12l2 2 4-4m5.618-4.016A11.955 11.955 0 0112 2.944a11.955 11.955 0 01-8.618 3.04A12.02 12.02 0 003 9c0 5.591 3.824 10.29 9 11.622 5.176-1.332 9-6.03 9-11.622 0-1.042-.133-2.052-.382-3.016z"/>
                                    </svg>
                                    <span class="text-xs mt-1">Protected</span>
                                </div>
                                <div class="flex flex-col items-center">
                                    <svg class="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                        <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M3 10h18M7 15h1m4 0h1m-7 4h12a3 3 0 003-3V8a3 3 0 00-3-3H6a3 3 0 00-3 3v8a3 3 0 003 3z"/>
                                    </svg>
                                    <span class="text-xs mt-1">Easy Pay</span>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        `,F(),fe(),v(q,R,g)}function I(g){let $=g.product||{},N=g.variant,M=$.slug||g.product_slug||"#",q=$.name||g.product_name||"Product",R=g.product_image||$.primary_image||$.image,X=$.id||g.product_id||"",G=L(g.price_at_add),W=L(g.current_price||g.unit_price),ne=L(g.total||g.line_total)||W*(g.quantity||1),re=G>W&&W>0;return`
            <div class="cart-item p-6 flex gap-4" data-item-id="${g.id}" data-product-id="${X}">
                <div class="flex-shrink-0 w-24 h-24 bg-gray-100 dark:bg-stone-700 rounded-lg overflow-hidden">
                    <a href="/products/${M}/">
                        ${R?`
                        <img 
                            src="${R}" 
                            alt="${Templates.escapeHtml(q)}"
                            class="w-full h-full object-cover"
                            onerror="this.style.display='none'"
                        >`:`
                        <div class="w-full h-full flex items-center justify-center text-gray-400 dark:text-stone-500">
                            <svg class="w-8 h-8" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="1.5" d="M4 16l4.586-4.586a2 2 0 012.828 0L16 16m-2-2l1.586-1.586a2 2 0 012.828 0L20 14m-6-6h.01M6 20h12a2 2 0 002-2V6a2 2 0 00-2-2H6a2 2 0 00-2 2v12a2 2 0 002 2z"/>
                            </svg>
                        </div>`}
                    </a>
                </div>
                <div class="flex-1 min-w-0">
                    <div class="flex justify-between">
                        <div>
                            <h3 class="font-medium text-gray-900 dark:text-white">
                                <a href="/products/${M}/" class="hover:text-primary-600 dark:hover:text-amber-400">
                                    ${Templates.escapeHtml(q)}
                                </a>
                            </h3>
                            ${N||g.variant_name?`
                                <p class="text-sm text-gray-500 dark:text-stone-400 mt-1">${Templates.escapeHtml(N?.name||N?.value||g.variant_name)}</p>
                            `:""}
                        </div>
                        <div class="flex items-center gap-2">
                            <button class="save-for-later-btn text-gray-400 dark:text-stone-500 hover:text-primary-600 dark:hover:text-amber-400 transition-colors" aria-label="Save for later" title="Save for later">
                                <svg class="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                    <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M4.318 6.318a4.5 4.5 0 000 6.364L12 20.364l7.682-7.682a4.5 4.5 0 00-6.364-6.364L12 7.636l-1.318-1.318a4.5 4.5 0 00-6.364 0z"/>
                                </svg>
                            </button>
                            <button class="remove-item-btn text-gray-400 dark:text-stone-500 hover:text-red-500 transition-colors" aria-label="Remove item">
                                <svg class="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                    <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M19 7l-.867 12.142A2 2 0 0116.138 21H7.862a2 2 0 01-1.995-1.858L5 7m5 4v6m4-6v6m1-10V4a1 1 0 00-1-1h-4a1 1 0 00-1 1v3M4 7h16"/>
                                </svg>
                            </button>
                        </div>
                    </div>
                    <div class="mt-4 flex items-center justify-between">
                        <div class="flex items-center border border-gray-300 dark:border-stone-600 rounded-lg">
                            <button 
                                class="qty-decrease px-3 py-1 text-gray-600 dark:text-stone-400 hover:text-gray-900 dark:hover:text-white hover:bg-gray-100 dark:hover:bg-stone-700 transition-colors"
                                aria-label="Decrease quantity"
                            >
                                <svg class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                    <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M20 12H4"/>
                                </svg>
                            </button>
                            <input 
                                type="text"
                                inputmode="numeric"
                                readonly
                                class="item-quantity w-12 text-center border-0 bg-transparent dark:text-white focus:ring-0 text-sm appearance-none"
                                value="${g.quantity}" 
                                min="1" 
                                max="${$.stock_quantity||99}"
                            >
                            <button 
                                class="qty-increase px-3 py-1 text-gray-600 dark:text-stone-400 hover:text-gray-900 dark:hover:text-white hover:bg-gray-100 dark:hover:bg-stone-700 transition-colors"
                                aria-label="Increase quantity"
                            >
                                <svg class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                    <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 4v16m8-8H4"/>
                                </svg>
                            </button>
                        </div>
                        <div class="text-right">
                            ${re?`
                                <span class="text-sm text-gray-400 dark:text-stone-500 line-through">${A(G*g.quantity)}</span>
                            `:""}
                            <span class="font-semibold text-gray-900 dark:text-white block">${A(ne)}</span>
                        </div>
                    </div>
                </div>
            </div>
        `}function F(){let g=document.getElementById("cart-items"),$=document.getElementById("clear-cart-btn"),N=document.getElementById("remove-coupon-btn");g&&g.getAttribute(w)!=="true"&&(g.setAttribute(w,"true"),g.addEventListener("click",async M=>{let q=M.target.closest(".cart-item");if(!q)return;let R=q.dataset.itemId,X=q.dataset.productId,G=q.querySelector(".item-quantity")||q.querySelector(".qty-input");if(M.target.closest(".remove-item-btn"))await K(R);else if(M.target.closest(".save-for-later-btn"))await ae(R,X,q);else if(M.target.closest(".qty-decrease")){let W=parseInt(G?.value,10)||1;W>1&&await Q(R,W-1)}else if(M.target.closest(".qty-increase")){let W=parseInt(G?.value,10)||1,ne=parseInt(G?.max,10)||99;W<ne&&await Q(R,W+1)}}),g.addEventListener("change",async M=>{if(M.target.classList.contains("item-quantity")||M.target.classList.contains("qty-input")){let R=M.target.closest(".cart-item")?.dataset.itemId,X=parseInt(M.target.value,10)||1;R&&X>0&&await Q(R,X)}})),$&&$.getAttribute(w)!=="true"&&($.setAttribute(w,"true"),$.addEventListener("click",async()=>{await Modal.confirm({title:"Clear Cart",message:"Are you sure you want to remove all items from your cart?",confirmText:"Clear Cart",cancelText:"Cancel"})&&await ge()})),N&&N.getAttribute(w)!=="true"&&(N.setAttribute(w,"true"),N.addEventListener("click",async()=>{await Be()}))}async function Q(g,$){try{await CartApi.updateItem(g,$),await l(),document.dispatchEvent(new CustomEvent("cart:updated"))}catch(N){Toast.error(N.message||"Failed to update quantity.")}}async function K(g){try{await CartApi.removeItem(g),Toast.success("Item removed from cart."),await l(),document.dispatchEvent(new CustomEvent("cart:updated"))}catch($){Toast.error($.message||"Failed to remove item.")}}async function ae(g,$,N){try{let M=n?.items?.find(G=>String(G.id)===String(g));if(!M)return;let q=M.product||{},R={id:$||q.id||M.product_id,name:q.name||M.product_name||"Product",image:M.product_image||q.primary_image||q.image||"",price:M.current_price||M.unit_price||q.price||0};e.findIndex(G=>G.id===$)===-1&&(e.push(R),localStorage.setItem("savedForLater",JSON.stringify(e))),await CartApi.removeItem(g),Toast.success("Item saved for later"),await l(),document.dispatchEvent(new CustomEvent("cart:updated"))}catch(M){Toast.error(M.message||"Failed to save item.")}}async function ge(){try{await CartApi.clearCart(),Toast.success("Cart cleared."),await l(),document.dispatchEvent(new CustomEvent("cart:updated"))}catch(g){Toast.error(g.message||"Failed to clear cart.")}}function Te(){let g=document.getElementById("coupon-form");g?.addEventListener("submit",async N=>{N.preventDefault();let q=document.getElementById("coupon-code")?.value.trim();if(!q){Toast.error("Please enter a coupon code.");return}let R=g.querySelector('button[type="submit"]');R.disabled=!0,R.textContent="Applying...";try{let X=L(n?.summary?.subtotal??n?.subtotal??0),G=await CartApi.applyCoupon(q,{subtotal:X}),W=G?.data?.cart?.coupon?.code||G?.data?.cart?.coupon_code||q;le(W),Toast.success("Coupon applied!"),await l()}catch(X){Toast.error(J(X,"Invalid coupon code."))}finally{R.disabled=!1,R.textContent="Apply"}});let $=document.getElementById("apply-coupon-btn");$?.addEventListener("click",async()=>{let M=document.getElementById("coupon-code")?.value.trim();if(!M){Toast.error("Please enter a coupon code.");return}$.disabled=!0;let q=$.textContent;$.textContent="Applying...";try{let R=L(n?.summary?.subtotal??n?.subtotal??0),X=await CartApi.applyCoupon(M,{subtotal:R}),G=X?.data?.cart?.coupon?.code||X?.data?.cart?.coupon_code||M;le(G),Toast.success("Coupon applied!"),await l()}catch(R){Toast.error(J(R,"Invalid coupon code."))}finally{$.disabled=!1,$.textContent=q||"Apply"}})}async function Be(){try{await CartApi.removeCoupon(),le(""),Toast.success("Coupon removed."),await l()}catch(g){Toast.error(J(g,"Failed to remove coupon."))}}function qe(){n=null}return{init:P,destroy:qe}})();window.CartPage=Ws;Nr=Ws});var Js={};te(Js,{default:()=>Fr});var Qs,Fr,Xs=ee(()=>{Qs=(function(){"use strict";let n={},e=1,s=null,o=null,h=!1,w=!1,_=!0,L=[],A=[],Y=4;async function Z(){if(h)return;h=!0;let t=D();if(!t)return;let r=document.getElementById("category-header");if(r&&r.querySelector("h1")){Ce(),_e(),$e(),z();return}n=J(),e=parseInt(new URLSearchParams(window.location.search).get("page"))||1,await le(t),Ce(),_e(),$e(),z()}function z(){E(),B(),i(),c(),m(),v()}function E(){let t=document.getElementById("load-more-trigger");if(!t)return;new IntersectionObserver(d=>{d.forEach(l=>{l.isIntersecting&&!w&&_&&H()})},{rootMargin:"200px 0px",threshold:.01}).observe(t)}async function H(){if(w||!_||!s)return;w=!0,e++;let t=document.getElementById("loading-more-indicator");t&&t.classList.remove("hidden");try{let r={category:s.id,page:e,limit:12,...n},d=await ProductsApi.getAll(r),l=d.data||[],p=d.meta||{};l.length===0?_=!1:(L=[...L,...l],U(l),_=e<(p.total_pages||1)),xe()}catch(r){console.error("Failed to load more products:",r)}finally{w=!1,t&&t.classList.add("hidden")}}function U(t){let r=document.getElementById("products-grid");if(!r)return;let d=Storage.get("productViewMode")||"grid";t.forEach(l=>{let p=ProductCard.render(l,{layout:d,showCompare:!0,showQuickView:!0}),f=document.createElement("div");f.innerHTML=p;let I=f.firstElementChild;I.classList.add("animate-fadeInUp"),r.appendChild(I)}),ProductCard.bindEvents(r)}function B(){A=JSON.parse(localStorage.getItem("compareProducts")||"[]"),k(),document.addEventListener("click",t=>{let r=t.target.closest("[data-compare]");if(!r)return;t.preventDefault();let d=parseInt(r.dataset.compare);C(d)})}function C(t){let r=A.findIndex(d=>d.id===t);if(r>-1)A.splice(r,1),Toast.info("Removed from compare");else{if(A.length>=Y){Toast.warning(`You can compare up to ${Y} products`);return}let d=L.find(l=>l.id===t);d&&(A.push({id:d.id,name:d.name,image:d.primary_image||d.image,price:d.price,sale_price:d.sale_price}),Toast.success("Added to compare"))}localStorage.setItem("compareProducts",JSON.stringify(A)),k(),S()}function k(){let t=document.getElementById("compare-bar");if(A.length===0){t?.remove();return}t||(t=document.createElement("div"),t.id="compare-bar",t.className="fixed bottom-0 left-0 right-0 bg-white dark:bg-stone-800 border-t border-stone-200 dark:border-stone-700 shadow-2xl z-40 transform transition-transform duration-300",document.body.appendChild(t)),t.innerHTML=`
            <div class="container mx-auto px-4 py-4">
                <div class="flex items-center justify-between gap-4">
                    <div class="flex items-center gap-3 overflow-x-auto">
                        <span class="text-sm font-medium text-stone-600 dark:text-stone-400 whitespace-nowrap">Compare (${A.length}/${Y}):</span>
                        ${A.map(r=>`
                            <div class="relative flex-shrink-0 group">
                                <img src="${r.image||"/static/images/placeholder.jpg"}" alt="${Templates.escapeHtml(r.name)}" class="w-14 h-14 object-cover rounded-lg border border-stone-200 dark:border-stone-600">
                                <button data-remove-compare="${r.id}" class="absolute -top-2 -right-2 w-5 h-5 bg-red-500 text-white rounded-full flex items-center justify-center opacity-0 group-hover:opacity-100 transition-opacity">
                                    <svg class="w-3 h-3" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M6 18L18 6M6 6l12 12"/></svg>
                                </button>
                            </div>
                        `).join("")}
                    </div>
                    <div class="flex items-center gap-2">
                        <button id="compare-now-btn" class="px-4 py-2 bg-primary-600 dark:bg-amber-600 text-white font-semibold rounded-lg hover:bg-primary-700 dark:hover:bg-amber-700 transition-colors disabled:opacity-50 disabled:cursor-not-allowed" ${A.length<2?"disabled":""}>
                            Compare Now
                        </button>
                        <button id="clear-compare-btn" class="px-4 py-2 text-stone-600 dark:text-stone-400 hover:text-stone-800 dark:hover:text-stone-200 transition-colors">
                            Clear All
                        </button>
                    </div>
                </div>
            </div>
        `,t.querySelectorAll("[data-remove-compare]").forEach(r=>{r.addEventListener("click",()=>{let d=parseInt(r.dataset.removeCompare);C(d)})}),document.getElementById("compare-now-btn")?.addEventListener("click",y),document.getElementById("clear-compare-btn")?.addEventListener("click",T)}function S(){document.querySelectorAll("[data-compare]").forEach(t=>{let r=parseInt(t.dataset.compare);A.some(l=>l.id===r)?(t.classList.add("bg-primary-100","text-primary-600"),t.classList.remove("bg-stone-100","text-stone-600")):(t.classList.remove("bg-primary-100","text-primary-600"),t.classList.add("bg-stone-100","text-stone-600"))})}function T(){A=[],localStorage.removeItem("compareProducts"),k(),S(),Toast.info("Compare list cleared")}async function y(){if(A.length<2)return;let t=document.createElement("div");t.id="compare-modal",t.className="fixed inset-0 z-50 overflow-auto",t.innerHTML=`
            <div class="absolute inset-0 bg-black/50 backdrop-blur-sm" onclick="document.getElementById('compare-modal').remove()"></div>
            <div class="relative min-h-full flex items-center justify-center p-4">
                <div class="bg-white dark:bg-stone-800 rounded-2xl shadow-2xl max-w-6xl w-full max-h-[90vh] overflow-auto">
                    <div class="sticky top-0 bg-white dark:bg-stone-800 border-b border-stone-200 dark:border-stone-700 p-4 flex items-center justify-between z-10">
                        <h2 class="text-xl font-bold text-stone-900 dark:text-white">Compare Products</h2>
                        <button onclick="document.getElementById('compare-modal').remove()" class="w-10 h-10 bg-stone-100 dark:bg-stone-700 rounded-full flex items-center justify-center hover:bg-stone-200 dark:hover:bg-stone-600 transition-colors">
                            <svg class="w-5 h-5 text-stone-600 dark:text-stone-300" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M6 18L18 6M6 6l12 12"/></svg>
                        </button>
                    </div>
                    <div class="p-4 overflow-x-auto">
                        <table class="w-full min-w-[600px]">
                            <thead>
                                <tr>
                                    <th class="text-left p-3 text-sm font-medium text-stone-500 dark:text-stone-400 w-32">Feature</th>
                                    ${A.map(r=>`
                                        <th class="p-3 text-center">
                                            <div class="flex flex-col items-center">
                                                <img src="${r.image||"/static/images/placeholder.jpg"}" alt="${Templates.escapeHtml(r.name)}" class="w-24 h-24 object-cover rounded-xl mb-2">
                                                <span class="text-sm font-semibold text-stone-900 dark:text-white">${Templates.escapeHtml(r.name)}</span>
                                            </div>
                                        </th>
                                    `).join("")}
                                </tr>
                            </thead>
                            <tbody>
                                <tr class="border-t border-stone-100 dark:border-stone-700">
                                    <td class="p-3 text-sm font-medium text-stone-600 dark:text-stone-400">Price</td>
                                    ${A.map(r=>`
                                        <td class="p-3 text-center">
                                            ${r.sale_price?`
                                                <span class="text-lg font-bold text-primary-600 dark:text-amber-400">${Templates.formatPrice(r.sale_price)}</span>
                                                <span class="text-sm text-stone-400 line-through ml-1">${Templates.formatPrice(r.price)}</span>
                                            `:`
                                                <span class="text-lg font-bold text-stone-900 dark:text-white">${Templates.formatPrice(r.price)}</span>
                                            `}
                                        </td>
                                    `).join("")}
                                </tr>
                                <tr class="border-t border-stone-100 dark:border-stone-700">
                                    <td class="p-3 text-sm font-medium text-stone-600 dark:text-stone-400">Actions</td>
                                    ${A.map(r=>`
                                        <td class="p-3 text-center">
                                            <button onclick="CartApi.addItem(${r.id}, 1).then(() => Toast.success('Added to cart'))" class="px-4 py-2 bg-primary-600 dark:bg-amber-600 text-white text-sm font-medium rounded-lg hover:bg-primary-700 dark:hover:bg-amber-700 transition-colors">
                                                Add to Cart
                                            </button>
                                        </td>
                                    `).join("")}
                                </tr>
                            </tbody>
                        </table>
                    </div>
                </div>
            </div>
        `,document.body.appendChild(t)}function i(){document.addEventListener("click",async t=>{let r=t.target.closest("[data-quick-view]");if(!r)return;let d=r.dataset.quickView;d&&(t.preventDefault(),await a(d))})}async function a(t){let r=document.createElement("div");r.id="quick-view-modal",r.className="fixed inset-0 z-50 flex items-center justify-center p-4",r.innerHTML=`
            <div class="absolute inset-0 bg-black/50 backdrop-blur-sm" onclick="document.getElementById('quick-view-modal').remove()"></div>
            <div class="relative bg-white dark:bg-stone-800 rounded-2xl shadow-2xl max-w-4xl w-full max-h-[90vh] overflow-auto">
                <div class="p-8 flex items-center justify-center min-h-[400px]">
                    <div class="animate-spin rounded-full h-12 w-12 border-b-2 border-primary-600 dark:border-amber-400"></div>
                </div>
            </div>
        `,document.body.appendChild(r);try{let d=await ProductsApi.getProduct(t),l=d.data||d,p=r.querySelector(".relative");p.innerHTML=`
                <button class="absolute top-4 right-4 z-10 w-10 h-10 bg-stone-100 dark:bg-stone-700 rounded-full flex items-center justify-center hover:bg-stone-200 dark:hover:bg-stone-600 transition-colors" onclick="document.getElementById('quick-view-modal').remove()">
                    <svg class="w-5 h-5 text-stone-600 dark:text-stone-300" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M6 18L18 6M6 6l12 12"/></svg>
                </button>
                <div class="grid md:grid-cols-2 gap-8 p-8">
                    <div>
                        <div class="aspect-square rounded-xl overflow-hidden bg-stone-100 dark:bg-stone-700 mb-4">
                            <img src="${l.primary_image||l.image||"/static/images/placeholder.jpg"}" alt="${Templates.escapeHtml(l.name)}" class="w-full h-full object-cover" id="quick-view-main-image">
                        </div>
                        ${l.images&&l.images.length>1?`
                            <div class="flex gap-2 overflow-x-auto pb-2">
                                ${l.images.slice(0,5).map((K,ae)=>`
                                    <button class="w-16 h-16 flex-shrink-0 rounded-lg overflow-hidden border-2 ${ae===0?"border-primary-600 dark:border-amber-400":"border-transparent"} hover:border-primary-400 transition-colors" onclick="document.getElementById('quick-view-main-image').src='${K.image||K}'">
                                        <img src="${K.thumbnail||K.image||K}" alt="" class="w-full h-full object-cover">
                                    </button>
                                `).join("")}
                            </div>
                        `:""}
                    </div>
                    <div class="flex flex-col">
                        <h2 class="text-2xl font-bold text-stone-900 dark:text-white mb-2">${Templates.escapeHtml(l.name)}</h2>
                        <div class="flex items-center gap-2 mb-4">
                            <div class="flex text-amber-400">
                                ${'<svg class="w-4 h-4" fill="currentColor" viewBox="0 0 20 20"><path d="M9.049 2.927c.3-.921 1.603-.921 1.902 0l1.286 3.967a1 1 0 00.95.69h4.178c.969 0 1.371 1.24.588 1.81l-3.385 2.46a1 1 0 00-.364 1.118l1.287 3.966c.3.922-.755 1.688-1.54 1.118l-3.385-2.46a1 1 0 00-1.175 0l-3.385 2.46c-.784.57-1.838-.196-1.54-1.118l1.287-3.966a1 1 0 00-.364-1.118l-3.385-2.46c-.783-.57-.38-1.81.588-1.81h4.178a1 1 0 00.95-.69l1.286-3.967z"/></svg>'.repeat(Math.round(l.rating||4))}
                            </div>
                            <span class="text-sm text-stone-500 dark:text-stone-400">(${l.review_count||0} reviews)</span>
                            ${l.stock_quantity<=5&&l.stock_quantity>0?`
                                <span class="text-sm text-orange-600 dark:text-orange-400 font-medium">Only ${l.stock_quantity} left!</span>
                            `:""}
                        </div>
                        <div class="mb-6">
                            ${l.sale_price||l.discounted_price?`
                                <span class="text-3xl font-bold text-primary-600 dark:text-amber-400">${Templates.formatPrice(l.sale_price||l.discounted_price)}</span>
                                <span class="text-lg text-stone-400 line-through ml-2">${Templates.formatPrice(l.price)}</span>
                                <span class="ml-2 px-2 py-1 bg-red-100 dark:bg-red-900/30 text-red-600 dark:text-red-400 text-sm font-medium rounded">Save ${Math.round((1-(l.sale_price||l.discounted_price)/l.price)*100)}%</span>
                            `:`
                                <span class="text-3xl font-bold text-stone-900 dark:text-white">${Templates.formatPrice(l.price)}</span>
                            `}
                        </div>
                        <p class="text-stone-600 dark:text-stone-400 mb-6 line-clamp-3">${Templates.escapeHtml(l.short_description||l.description||"")}</p>
                        
                        <!-- Quantity Selector -->
                        <div class="flex items-center gap-4 mb-6">
                            <span class="text-sm font-medium text-stone-700 dark:text-stone-300">Quantity:</span>
                            <div class="flex items-center border border-stone-300 dark:border-stone-600 rounded-lg">
                                <button id="qv-qty-minus" class="w-10 h-10 flex items-center justify-center text-stone-600 dark:text-stone-400 hover:bg-stone-100 dark:hover:bg-stone-700 transition-colors">\u2212</button>
                                <input type="number" id="qv-qty-input" value="1" min="1" max="${l.stock_quantity||99}" class="w-16 h-10 text-center border-x border-stone-300 dark:border-stone-600 bg-transparent text-stone-900 dark:text-white">
                                <button id="qv-qty-plus" class="w-10 h-10 flex items-center justify-center text-stone-600 dark:text-stone-400 hover:bg-stone-100 dark:hover:bg-stone-700 transition-colors">+</button>
                            </div>
                        </div>

                        <div class="mt-auto space-y-3">
                            <button id="qv-add-to-cart" class="w-full py-3 px-6 bg-primary-600 dark:bg-amber-600 hover:bg-primary-700 dark:hover:bg-amber-700 text-white font-semibold rounded-xl transition-colors flex items-center justify-center gap-2">
                                <svg class="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M3 3h2l.4 2M7 13h10l4-8H5.4M7 13L5.4 5M7 13l-2.293 2.293c-.63.63-.184 1.707.707 1.707H17m0 0a2 2 0 100 4 2 2 0 000-4zm-8 2a2 2 0 11-4 0 2 2 0 014 0z"/></svg>
                                Add to Cart
                            </button>
                            <div class="grid grid-cols-2 gap-3">
                                <button onclick="WishlistApi.add(${l.id}).then(() => Toast.success('Added to wishlist'))" class="py-3 px-6 border-2 border-stone-200 dark:border-stone-600 text-stone-700 dark:text-stone-300 font-semibold rounded-xl hover:bg-stone-50 dark:hover:bg-stone-700 transition-colors flex items-center justify-center gap-2">
                                    <svg class="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M4.318 6.318a4.5 4.5 0 000 6.364L12 20.364l7.682-7.682a4.5 4.5 0 00-6.364-6.364L12 7.636l-1.318-1.318a4.5 4.5 0 00-6.364 0z"/></svg>
                                    Wishlist
                                </button>
                                <a href="/products/${l.slug||l.id}/" class="py-3 px-6 border-2 border-stone-200 dark:border-stone-600 text-stone-700 dark:text-stone-300 font-semibold rounded-xl text-center hover:bg-stone-50 dark:hover:bg-stone-700 transition-colors">
                                    Full Details
                                </a>
                            </div>
                        </div>
                    </div>
                </div>
            `;let f=document.getElementById("qv-qty-input"),I=document.getElementById("qv-qty-minus"),F=document.getElementById("qv-qty-plus"),Q=document.getElementById("qv-add-to-cart");I?.addEventListener("click",()=>{let K=parseInt(f.value)||1;K>1&&(f.value=K-1)}),F?.addEventListener("click",()=>{let K=parseInt(f.value)||1,ae=parseInt(f.max)||99;K<ae&&(f.value=K+1)}),Q?.addEventListener("click",async()=>{let K=parseInt(f.value)||1;Q.disabled=!0,Q.innerHTML='<svg class="animate-spin h-5 w-5 mx-auto" fill="none" viewBox="0 0 24 24"><circle class="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" stroke-width="4"></circle><path class="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path></svg>';try{await CartApi.addItem(l.id,K),Toast.success("Added to cart"),r.remove()}catch{Toast.error("Failed to add to cart")}finally{Q.disabled=!1,Q.innerHTML='<svg class="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M3 3h2l.4 2M7 13h10l4-8H5.4M7 13L5.4 5M7 13l-2.293 2.293c-.63.63-.184 1.707.707 1.707H17m0 0a2 2 0 100 4 2 2 0 000-4zm-8 2a2 2 0 11-4 0 2 2 0 014 0z"/></svg> Add to Cart'}})}catch(d){console.error("Failed to load product:",d),r.remove(),Toast.error("Failed to load product details")}}function c(){if(!document.getElementById("price-range-slider"))return;let r=document.getElementById("filter-min-price"),d=document.getElementById("filter-max-price");!r||!d||[r,d].forEach(l=>{l.addEventListener("input",()=>{u()})})}function u(){let t=document.getElementById("price-range-display"),r=document.getElementById("filter-min-price")?.value||0,d=document.getElementById("filter-max-price")?.value||"\u221E";t&&(t.textContent=`$${r} - $${d}`)}function m(){b()}function b(){let t=document.getElementById("active-filters");if(!t)return;let r=[];if(n.min_price&&r.push({key:"min_price",label:`Min: $${n.min_price}`}),n.max_price&&r.push({key:"max_price",label:`Max: $${n.max_price}`}),n.in_stock&&r.push({key:"in_stock",label:"In Stock"}),n.on_sale&&r.push({key:"on_sale",label:"On Sale"}),n.ordering){let d={price:"Price: Low to High","-price":"Price: High to Low","-created_at":"Newest First",name:"A-Z","-popularity":"Most Popular"};r.push({key:"ordering",label:d[n.ordering]||n.ordering})}if(r.length===0){t.innerHTML="";return}t.innerHTML=`
            <div class="flex flex-wrap items-center gap-2 mb-4">
                <span class="text-sm text-stone-500 dark:text-stone-400">Active filters:</span>
                ${r.map(d=>`
                    <button data-remove-filter="${d.key}" class="inline-flex items-center gap-1 px-3 py-1 bg-primary-100 dark:bg-amber-900/30 text-primary-700 dark:text-amber-400 rounded-full text-sm hover:bg-primary-200 dark:hover:bg-amber-900/50 transition-colors">
                        ${d.label}
                        <svg class="w-3 h-3" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M6 18L18 6M6 6l12 12"/></svg>
                    </button>
                `).join("")}
                <button id="clear-all-active-filters" class="text-sm text-stone-500 dark:text-stone-400 hover:text-stone-700 dark:hover:text-stone-300 underline">Clear all</button>
            </div>
        `,t.querySelectorAll("[data-remove-filter]").forEach(d=>{d.addEventListener("click",()=>{let l=d.dataset.removeFilter;delete n[l],ce()})}),document.getElementById("clear-all-active-filters")?.addEventListener("click",()=>{n={},ce()})}function v(){P()}function P(t=null){let r=document.getElementById("product-count");r&&t!==null&&(r.textContent=`${t} products`)}function D(){let r=window.location.pathname.match(/\/categories\/([^\/]+)/);return r?r[1]:null}function J(){let t=new URLSearchParams(window.location.search),r={};t.get("min_price")&&(r.min_price=t.get("min_price")),t.get("max_price")&&(r.max_price=t.get("max_price")),t.get("ordering")&&(r.ordering=t.get("ordering")),t.get("in_stock")&&(r.in_stock=t.get("in_stock")==="true"),t.get("on_sale")&&(r.on_sale=t.get("on_sale")==="true");let d=t.getAll("attr");return d.length&&(r.attributes=d),r}async function le(t){let r=document.getElementById("category-header"),d=document.getElementById("category-products"),l=document.getElementById("category-filters");r&&Loader.show(r,"skeleton"),d&&Loader.show(d,"skeleton");try{let p=await CategoriesApi.getCategory(t);if(s=p.data||p,!s){window.location.href="/404/";return}be(s),await ke(s),await fe(s),await pe(),await me(s)}catch(p){console.error("Failed to load category:",p),r&&(r.innerHTML='<p class="text-red-500">Failed to load category.</p>')}}function be(t){let r=document.getElementById("category-header");r&&(document.title=`${t.name} | Bunoraa`,r.innerHTML=`
            <div class="relative py-8 md:py-12">
                ${t.image?`
                    <div class="absolute inset-0 overflow-hidden rounded-2xl">
                        <img src="${t.image}" alt="" class="w-full h-full object-cover opacity-20">
                        <div class="absolute inset-0 bg-gradient-to-r from-white via-white/95 to-white/80"></div>
                    </div>
                `:""}
                <div class="relative">
                    <h1 class="text-3xl md:text-4xl font-bold text-gray-900 mb-2">${Templates.escapeHtml(t.name)}</h1>
                    ${t.description?`
                        <p class="text-gray-600 max-w-2xl">${Templates.escapeHtml(t.description)}</p>
                    `:""}
                    ${t.product_count?`
                        <p class="mt-4 text-sm text-gray-500">${t.product_count} products</p>
                    `:""}
                </div>
            </div>
        `)}async function ke(t){let r=document.getElementById("breadcrumbs");if(r)try{let l=(await CategoriesApi.getBreadcrumbs(t.id)).data||[],p=[{label:"Home",url:"/"},{label:"Categories",url:"/categories/"},...l.map(f=>({label:f.name,url:`/categories/${f.slug}/`}))];r.innerHTML=Breadcrumb.render(p)}catch(d){console.error("Failed to load breadcrumbs:",d)}}async function fe(t){let r=document.getElementById("category-filters");if(r)try{let l=(await ProductsApi.getFilterOptions({category:t.id})).data||{};r.innerHTML=`
                <div class="space-y-6">
                    <!-- Price Range -->
                    <div class="border-b border-gray-200 pb-6">
                        <h3 class="text-sm font-semibold text-gray-900 mb-4">Price Range</h3>
                        <div class="flex items-center gap-2">
                            <input 
                                type="number" 
                                id="filter-min-price" 
                                placeholder="Min"
                                value="${n.min_price||""}"
                                class="w-full px-3 py-2 border border-gray-300 rounded-lg text-sm focus:ring-primary-500 focus:border-primary-500"
                            >
                            <span class="text-gray-400">-</span>
                            <input 
                                type="number" 
                                id="filter-max-price" 
                                placeholder="Max"
                                value="${n.max_price||""}"
                                class="w-full px-3 py-2 border border-gray-300 rounded-lg text-sm focus:ring-primary-500 focus:border-primary-500"
                            >
                        </div>
                        <button id="apply-price-filter" class="mt-3 w-full px-4 py-2 bg-gray-100 hover:bg-gray-200 text-gray-700 text-sm font-medium rounded-lg transition-colors">
                            Apply
                        </button>
                    </div>

                    <!-- Availability -->
                    <div class="border-b border-gray-200 pb-6">
                        <h3 class="text-sm font-semibold text-gray-900 mb-4">Availability</h3>
                        <div class="space-y-2">
                            <label class="flex items-center">
                                <input 
                                    type="checkbox" 
                                    id="filter-in-stock"
                                    ${n.in_stock?"checked":""}
                                    class="w-4 h-4 text-primary-600 border-gray-300 rounded focus:ring-primary-500"
                                >
                                <span class="ml-2 text-sm text-gray-600">In Stock</span>
                            </label>
                            <label class="flex items-center">
                                <input 
                                    type="checkbox" 
                                    id="filter-on-sale"
                                    ${n.on_sale?"checked":""}
                                    class="w-4 h-4 text-primary-600 border-gray-300 rounded focus:ring-primary-500"
                                >
                                <span class="ml-2 text-sm text-gray-600">On Sale</span>
                            </label>
                        </div>
                    </div>

                    ${l.attributes&&l.attributes.length?`
                        ${l.attributes.map(p=>`
                            <div class="border-b border-gray-200 pb-6">
                                <h3 class="text-sm font-semibold text-gray-900 mb-4">${Templates.escapeHtml(p.name)}</h3>
                                <div class="space-y-2 max-h-48 overflow-y-auto">
                                    ${p.values.map(f=>`
                                        <label class="flex items-center">
                                            <input 
                                                type="checkbox" 
                                                name="attr-${p.slug}"
                                                value="${Templates.escapeHtml(f.value)}"
                                                ${n.attributes?.includes(`${p.slug}:${f.value}`)?"checked":""}
                                                class="filter-attribute w-4 h-4 text-primary-600 border-gray-300 rounded focus:ring-primary-500"
                                                data-attribute="${p.slug}"
                                            >
                                            <span class="ml-2 text-sm text-gray-600">${Templates.escapeHtml(f.value)}</span>
                                            ${f.count?`<span class="ml-auto text-xs text-gray-400">(${f.count})</span>`:""}
                                        </label>
                                    `).join("")}
                                </div>
                            </div>
                        `).join("")}
                    `:""}

                    <!-- Clear Filters -->
                    <button id="clear-filters" class="w-full px-4 py-2 border border-gray-300 text-gray-700 text-sm font-medium rounded-lg hover:bg-gray-50 transition-colors">
                        Clear All Filters
                    </button>
                </div>
            `,we()}catch(d){console.error("Failed to load filters:",d),r.innerHTML=""}}function we(){let t=document.getElementById("apply-price-filter"),r=document.getElementById("filter-in-stock"),d=document.getElementById("filter-on-sale"),l=document.getElementById("clear-filters"),p=document.querySelectorAll(".filter-attribute");t?.addEventListener("click",()=>{let f=document.getElementById("filter-min-price")?.value,I=document.getElementById("filter-max-price")?.value;f?n.min_price=f:delete n.min_price,I?n.max_price=I:delete n.max_price,ce()}),r?.addEventListener("change",f=>{f.target.checked?n.in_stock=!0:delete n.in_stock,ce()}),d?.addEventListener("change",f=>{f.target.checked?n.on_sale=!0:delete n.on_sale,ce()}),p.forEach(f=>{f.addEventListener("change",()=>{ye(),ce()})}),l?.addEventListener("click",()=>{n={},e=1,ce()})}function ye(){let t=document.querySelectorAll(".filter-attribute:checked"),r=[];t.forEach(d=>{r.push(`${d.dataset.attribute}:${d.value}`)}),r.length?n.attributes=r:delete n.attributes}function ce(){e=1,xe(),pe()}function xe(){let t=new URLSearchParams;n.min_price&&t.set("min_price",n.min_price),n.max_price&&t.set("max_price",n.max_price),n.ordering&&t.set("ordering",n.ordering),n.in_stock&&t.set("in_stock","true"),n.on_sale&&t.set("on_sale","true"),n.attributes&&n.attributes.forEach(d=>t.append("attr",d)),e>1&&t.set("page",e);let r=`${window.location.pathname}${t.toString()?"?"+t.toString():""}`;window.history.pushState({},"",r)}async function pe(){let t=document.getElementById("category-products");if(!(!t||!s)){o&&o.abort(),o=new AbortController,Loader.show(t,"skeleton");try{let r={category:s.id,page:e,limit:12,...n};n.attributes&&(delete r.attributes,n.attributes.forEach(f=>{let[I,F]=f.split(":");r[`attr_${I}`]=F}));let d=await ProductsApi.getAll(r),l=d.data||[],p=d.meta||{};L=l,_=e<(p.total_pages||1),Ee(l,p),b(),P(p.total||l.length)}catch(r){if(r.name==="AbortError")return;console.error("Failed to load products:",r),t.innerHTML='<p class="text-red-500 text-center py-8">Failed to load products. Please try again.</p>'}}}function Ee(t,r){let d=document.getElementById("category-products");if(!d)return;let l=Storage.get("productViewMode")||"grid",p=l==="list"?"space-y-4":"grid grid-cols-2 md:grid-cols-3 lg:grid-cols-4 gap-4 md:gap-6";if(t.length===0){d.innerHTML=`
                <div class="text-center py-12">
                    <svg class="w-16 h-16 text-gray-300 mx-auto mb-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9.172 16.172a4 4 0 015.656 0M9 10h.01M15 10h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z"/>
                    </svg>
                    <h3 class="text-lg font-medium text-gray-900 dark:text-white mb-2">No products found</h3>
                    <p class="text-gray-500 dark:text-stone-400 mb-4">Try adjusting your filters or browse other categories.</p>
                    <button id="clear-filters-empty" class="px-4 py-2 bg-primary-600 dark:bg-amber-600 text-white rounded-lg hover:bg-primary-700 dark:hover:bg-amber-700 transition-colors">
                        Clear Filters
                    </button>
                </div>
            `,document.getElementById("clear-filters-empty")?.addEventListener("click",()=>{n={},e=1,ce()});return}if(d.innerHTML=`
            <div id="active-filters" class="mb-4"></div>
            <div id="products-grid" class="${p}">
                ${t.map(f=>ProductCard.render(f,{layout:l,showCompare:!0,showQuickView:!0})).join("")}
            </div>
            
            <!-- Infinite Scroll Trigger -->
            <div id="load-more-trigger" class="h-20 flex items-center justify-center">
                <div id="loading-more-indicator" class="hidden">
                    <svg class="animate-spin h-8 w-8 text-primary-600 dark:text-amber-400" fill="none" viewBox="0 0 24 24">
                        <circle class="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" stroke-width="4"></circle>
                        <path class="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                    </svg>
                </div>
            </div>
            
            ${r.total_pages>1?`
                <div id="products-pagination" class="mt-8"></div>
            `:""}
        `,ProductCard.bindEvents(d),b(),S(),E(),r.total_pages>1){let f=document.getElementById("products-pagination");f.innerHTML=Pagination.render({currentPage:r.current_page||e,totalPages:r.total_pages,totalItems:r.total}),f.addEventListener("click",I=>{let F=I.target.closest("[data-page]");F&&(e=parseInt(F.dataset.page),_=!0,xe(),pe(),window.scrollTo({top:0,behavior:"smooth"}))})}}async function me(t){let r=document.getElementById("subcategories");if(r)try{let l=(await CategoriesApi.getSubcategories(t.id)).data||[];if(l.length===0){r.innerHTML="";return}r.innerHTML=`
                <div class="mb-8">
                    <h2 class="text-lg font-semibold text-gray-900 mb-4">Browse Subcategories</h2>
                    <div class="flex flex-wrap gap-2">
                        ${l.map(p=>`
                            <a href="/categories/${p.slug}/" class="px-4 py-2 bg-gray-100 hover:bg-gray-200 text-gray-700 rounded-full text-sm transition-colors">
                                ${Templates.escapeHtml(p.name)}
                                ${p.product_count?`<span class="text-gray-400 ml-1">(${p.product_count})</span>`:""}
                            </a>
                        `).join("")}
                    </div>
                </div>
            `}catch(d){console.error("Failed to load subcategories:",d),r.innerHTML=""}}function Ce(){let t=document.getElementById("mobile-filter-btn"),r=document.getElementById("filter-sidebar"),d=document.getElementById("close-filter-btn");t?.addEventListener("click",()=>{r?.classList.remove("hidden"),document.body.classList.add("overflow-hidden")}),d?.addEventListener("click",()=>{r?.classList.add("hidden"),document.body.classList.remove("overflow-hidden")})}function _e(){let t=document.getElementById("sort-select");t&&(t.value=n.ordering||"",t.addEventListener("change",r=>{r.target.value?n.ordering=r.target.value:delete n.ordering,ce()}))}function $e(){let t=document.getElementById("view-grid"),r=document.getElementById("view-list");(Storage.get("productViewMode")||"grid")==="list"&&(t?.classList.remove("bg-gray-200"),r?.classList.add("bg-gray-200")),t?.addEventListener("click",()=>{Storage.set("productViewMode","grid"),t.classList.add("bg-gray-200"),r?.classList.remove("bg-gray-200"),pe()}),r?.addEventListener("click",()=>{Storage.set("productViewMode","list"),r.classList.add("bg-gray-200"),t?.classList.remove("bg-gray-200"),pe()})}function x(){o&&(o.abort(),o=null),n={},e=1,s=null,h=!1,w=!1,_=!0,L=[],document.getElementById("compare-bar")?.remove(),document.getElementById("quick-view-modal")?.remove(),document.getElementById("compare-modal")?.remove()}return{init:Z,destroy:x,toggleCompare:C,clearCompare:T}})();window.CategoryPage=Qs;Fr=Qs});var Zs={};te(Zs,{default:()=>zr});var Ks,zr,er=ee(()=>{Ks=(async function(){"use strict";let n=null,e={shipping_address:null,billing_address:null,same_as_shipping:!0,shipping_method:null,payment_method:null,notes:""},s=1,o=!1,h=null,w=(x,t=0)=>{if(x==null||x==="")return t;let r=Number(x);return Number.isFinite(r)?r:t},_="checkout:shipping",L=1e3*60*60*24*7;function A(){try{let x=localStorage.getItem(_);if(!x)return null;let t=JSON.parse(x);if(!t||typeof t!="object"||t.ts&&Date.now()-t.ts>L)return null;if(t.cost!==void 0&&t.cost!==null){let r=Number(t.cost);if(!Number.isFinite(r))return null;t.cost=r}return t}catch{return null}}function Y(){return document.querySelector('input[name="shipping_rate_id"]:checked')||document.querySelector('input[name="shipping_method"]:checked')}function Z(x,t){let r=Y();if(r){let f=w(r.dataset.price,NaN);if(Number.isFinite(f))return{cost:f,display:r.dataset.display||null,currency:r.dataset.currency||null,source:"checked"}}let d=A();if(d&&Number.isFinite(Number(d.cost)))return{cost:Number(d.cost),display:d.display||null,currency:d.currency||null,source:"stored"};let l=x?.querySelector("#shipping-cost")||document.getElementById("shipping-cost"),p=l?.dataset?.price;if(p!==void 0&&p!==""){let f=w(p,NaN);if(Number.isFinite(f)){let I=l?.textContent?.trim()||"",F=I&&I.toLowerCase()!=="calculated next"?I:null;return{cost:f,display:F,currency:l?.dataset?.currency||null,source:"dom"}}}if(t&&t.shipping_cost!==void 0&&t.shipping_cost!==null){let f=w(t.shipping_cost,NaN);if(Number.isFinite(f)&&f>0)return{cost:f,display:null,currency:null,source:"cart"}}return null}function z(x){return!x||x.cost===null||Number.isNaN(x.cost)?"Calculated next":x.display&&String(x.display).trim()&&String(x.display).toLowerCase()!=="calculated next"?String(x.display):x.cost<=0?"Free":Templates.formatPrice(x.cost,x.currency||null)}function E(){let x=document.querySelector('input[name="payment_method"]:checked');if(x)return{type:x.dataset.feeType||"none",amount:w(x.dataset.feeAmount,0),percent:w(x.dataset.feePercent,0),name:x.dataset.feeName||""};let t=document.getElementById("order-summary");if(t){let r=w(t.dataset.paymentFee,0);return{type:r>0?"flat":"none",amount:r,percent:0,name:t.dataset.paymentFeeLabel||""}}return{type:"none",amount:0,percent:0,name:""}}function H(x,t){return t?t.type==="flat"?w(t.amount,0):t.type==="percent"?Math.max(0,x*(w(t.percent,0)/100)):0:0}function U(x=null){let t=document.getElementById("payment-fee-row"),r=document.getElementById("payment-fee-amount"),d=document.getElementById("payment-fee-label");if(!t||!r)return;let l=document.getElementById("order-total"),p=x!==null?x:w(l?.dataset?.price??l?.textContent,0),f=E(),I=H(p,f);if(!I||I<=0){t.classList.add("hidden");return}t.classList.remove("hidden"),r.textContent=Templates.formatPrice(I),d&&(d.textContent=f?.name?`Extra payment fee (${f.name})`:"Extra payment fee");let F=document.getElementById("order-summary");F&&(F.dataset.paymentFee=I,F.dataset.paymentFeeLabel=f?.name||"")}async function B(){if(!AuthApi.isAuthenticated()&&!document.getElementById("guest-checkout")){Toast.info("Please login to continue checkout."),window.location.href="/account/login/?next=/checkout/";return}if(await C(),!o){if(!n||!n.items||n.items.length===0){Toast.warning("Your cart is empty.");return}await a(),u(),pe(),ce(),Ee()}}async function C(){try{let x=await CartApi.getCart();if(!x||x.success===!1)throw{message:x?.message||"Failed to load cart",data:x?.data};n=x.data,i(),o=!1,h=null}catch(x){console.error("Failed to load cart:",x),o=!0,h=x,Toast.error(k(x,"Failed to load cart."))}}function k(x,t){let r=["request failed.","request failed","invalid response format","invalid request format"],d=I=>{if(!I)return!0;let F=String(I).trim().toLowerCase();return r.includes(F)},l=I=>{if(!I)return null;if(typeof I=="string")return I;if(Array.isArray(I))return I[0];if(typeof I=="object"){let F=Object.values(I),K=(F.flat?F.flat():F.reduce((ae,ge)=>ae.concat(ge),[]))[0]??F[0];if(typeof K=="string")return K;if(K&&typeof K=="object")return l(K)}return null},p=[];return x?.message&&p.push(x.message),x?.data?.message&&p.push(x.data.message),x?.data?.detail&&p.push(x.data.detail),x?.data&&typeof x.data=="string"&&p.push(x.data),x?.errors&&p.push(l(x.errors)),x?.data&&typeof x.data=="object"&&p.push(l(x.data)),p.find(I=>I&&!d(I))||t}function S(x){if(!x)return{};let t=x.querySelector('button[type="submit"]'),r=x.querySelector("#btn-text")||x.querySelector("#button-text"),d=x.querySelector("#btn-spinner")||x.querySelector("#spinner"),l=x.querySelector("#arrow-icon"),p=r?r.textContent:t?t.textContent:"";return{button:t,textEl:r,spinnerEl:d,arrowEl:l,defaultText:p}}function T(x,t,r="Processing..."){x&&(x.button&&(x.button.disabled=t),x.textEl&&(x.textEl.textContent=t?r:x.defaultText),x.spinnerEl&&x.spinnerEl.classList.toggle("hidden",!t),x.arrowEl&&x.arrowEl.classList.toggle("hidden",t))}async function y(x,t={}){if(!x||x.dataset.submitting==="true")return;if(o){Toast.error(k(h,"Failed to load cart."));return}let r=t.validate;if(typeof r=="function"&&!await r())return;let d=S(x);T(d,!0,t.loadingText||"Processing..."),x.dataset.submitting="true";try{let l=await fetch(x.action||window.location.href,{method:(x.method||"POST").toUpperCase(),body:new FormData(x),headers:{"X-Requested-With":"XMLHttpRequest"},credentials:"same-origin"}),p=null;try{p=await l.json()}catch{p=null}if(!l.ok||p&&p.success===!1)throw{message:p?.message||l.statusText||"Request failed.",data:p};let f=p?.redirect_url||t.redirectUrl;if(f){window.location.href=f;return}typeof t.onSuccess=="function"&&t.onSuccess(p)}catch(l){Toast.error(k(l,t.errorMessage||"Request failed.")),typeof t.onError=="function"&&t.onError(l)}finally{x.dataset.submitting="false",T(d,!1,t.loadingText||"Processing...")}}function i(){let x=document.getElementById("order-summary");if(!x||!n)return;let t=Array.isArray(n.items)?n.items:[],r=Number(n.item_count??t.length??0),d=`${r} item${r===1?"":"s"}`;document.querySelectorAll("[data-order-items-count]").forEach(de=>{de.textContent=d});let l=de=>{let Ae=Number(de);return Number.isFinite(Ae)?Ae:0},p=()=>{let de=document.getElementById("tax-rate-data")||document.querySelector("[data-tax-rate]");if(!de)return 0;let Ae=de.dataset?.taxRate??de.textContent??"",Me=parseFloat(Ae);return Number.isFinite(Me)?Me:0},f=de=>Templates.escapeHtml(String(de??"")),I=()=>t.length?t.map((de,Ae)=>{let Me=de.product?.name||de.product_name||de.name||"Item",kt=de.product?.image||de.product_image||de.image||null,Et=de.variant?.name||de.variant?.value||de.variant_name||"",Ct=l(de.quantity||0),$r=l(de.price??de.unit_price??de.unitPrice??de.price_at_add??0),Tr=l(de.total??$r*Ct);return`
                    <div class="flex items-start space-x-4 py-3 ${Ae!==t.length-1?"border-b border-gray-100 dark:border-gray-700":""}">
                        <div class="relative flex-shrink-0">
                            ${kt?`
                                <img src="${kt}" alt="${f(Me)}" class="w-16 h-16 object-cover rounded-lg" loading="lazy" decoding="async" onerror="this.style.display='none'">
                            `:`
                                <div class="w-16 h-16 bg-gray-200 dark:bg-gray-700 rounded-lg flex items-center justify-center text-gray-400">
                                    <svg class="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                        <path stroke-linecap="round" stroke-linejoin="round" stroke-width="1.5" d="M4 16l4.586-4.586a2 2 0 012.828 0L16 16m-2-2l1.586-1.586a2 2 0 012.828 0L20 14m-6-6h.01M6 20h12a2 2 0 002-2V6a2 2 0 00-2-2H6a2 2 0 00-2 2v12a2 2 0 002 2z"/>
                                    </svg>
                                </div>
                            `}
                            <span class="absolute -top-2 -right-2 w-5 h-5 bg-gray-600 text-white text-xs rounded-full flex items-center justify-center font-medium">
                                ${Ct}
                            </span>
                        </div>
                        <div class="flex-1 min-w-0">
                            <p class="text-sm font-medium text-gray-900 dark:text-white truncate">${f(Me)}</p>
                            ${Et?`<p class="text-xs text-gray-500 dark:text-gray-400">${f(Et)}</p>`:""}
                        </div>
                        <p class="text-sm font-medium text-gray-900 dark:text-white">${Templates.formatPrice(Tr)}</p>
                    </div>
                `}).join(""):'<p class="text-gray-500 dark:text-gray-400 text-center py-4">Your cart is empty</p>',F=l(n.subtotal),Q=l(n.discount_amount),K=n.total!==void 0&&n.total!==null?l(n.total):Math.max(0,F-Q),ae=Z(x,n),ge=p(),Be=(x.querySelector("#tax-amount")||document.getElementById("tax-amount"))?.dataset?.price,qe=Be!==void 0&&Be!==""?l(Be):null,g=n.tax_amount!==void 0&&n.tax_amount!==null?l(n.tax_amount):qe!==null?qe:ge>0?K*ge/100:0,$=x.querySelector("#gift-wrap-row")||document.getElementById("gift-wrap-row"),N=x.querySelector("#gift-wrap-cost")||document.getElementById("gift-wrap-cost"),M=$?.querySelector("span")?.textContent?.trim()||"Gift Wrapping",q=N?.dataset?.price??N?.textContent??x.dataset?.giftWrapCost??0,R=l(x.dataset?.giftWrapAmount??0),G=!!document.getElementById("gift_wrap")?.checked||$&&$.style.display!=="none"&&!$.classList.contains("hidden"),W=l(q);G&&W<=0&&R>0&&(W=R);let ne=G||W>0,re=ae&&ae.cost!==null&&!Number.isNaN(ae.cost),se=K+(re?ae.cost:0)+(g||0)+(ne?W:0),ue=re?z(ae):"Calculated next",ve=ae?.currency?` data-currency="${f(ae.currency)}"`:"",he=E(),Le=H(se,he);x&&(x.dataset.paymentFee=Le,he?.name&&(x.dataset.paymentFeeLabel=he.name),x.dataset.giftWrapCost=W,x.dataset.giftWrapAmount=R);let Ie=`
            <div class="space-y-4 max-h-80 overflow-y-auto scrollbar-thin pr-2">
                ${I()}
            </div>
        `,Lr=`
            <div id="payment-fee-row" class="flex justify-between text-sm text-gray-600 dark:text-gray-400 ${Le>0?"":"hidden"}">
                <span id="payment-fee-label">Extra payment fee${he?.name?` (${f(he.name)})`:""}</span>
                <span id="payment-fee-amount">${Templates.formatPrice(Le)}</span>
            </div>
        `,_r=`
            <div id="gift-wrap-row" class="flex justify-between text-sm text-gray-600 dark:text-gray-400" style="display: ${ne?"flex":"none"};">
                <span>${f(M)}</span>
                <span id="gift-wrap-cost" data-price="${W}">+${Templates.formatPrice(W)}</span>
            </div>
        `,wt=`
            <div class="space-y-3 border-t border-gray-200 dark:border-gray-700 mt-4 pt-4">
                <div class="flex justify-between text-sm text-gray-600 dark:text-gray-400">
                    <span>Subtotal</span>
                    <span id="subtotal" data-price="${F}">${Templates.formatPrice(F)}</span>
                </div>
                <div id="discount-row" class="flex justify-between text-sm text-green-600 ${Q>0?"":"hidden"}">
                    <span>Discount</span>
                    <span id="discount-amount" data-price="${Q}">-${Templates.formatPrice(Q)}</span>
                    <span id="discount" class="hidden" data-price="${Q}">-${Templates.formatPrice(Q)}</span>
                </div>
                <div class="flex justify-between text-sm text-gray-600 dark:text-gray-400">
                    <span>Shipping</span>
                    <span id="shipping-cost" data-price="${re?ae.cost:""}"${ve} class="font-medium text-gray-900 dark:text-white">
                        ${re?f(ue):"Calculated next"}
                    </span>
                </div>
                ${_r}
                ${Lr}
                <div class="flex justify-between text-sm text-gray-600 dark:text-gray-400 ${ge>0||g>0?"":"hidden"}">
                    <span>Tax${ge>0?` (${ge}%)`:""}</span>
                    <span id="tax-amount" data-price="${g}">${Templates.formatPrice(g)}</span>
                </div>
                <div class="flex justify-between text-lg font-bold text-gray-900 dark:text-white border-t border-gray-200 dark:border-gray-700 pt-3">
                    <span>Total</span>
                    <span id="order-total" data-price="${se}">${Templates.formatPrice(se)}</span>
                </div>
            </div>
        `,je=x.querySelector("[data-order-items]"),Ne=x.querySelector("[data-order-totals]");if(je||Ne){je&&(je.innerHTML=Ie),Ne&&(Ne.innerHTML=wt),U(se);return}x.innerHTML=Ie+wt,U(se)}async function a(){if(AuthApi.isAuthenticated())try{let t=(await AuthApi.getAddresses()).data||[],r=document.getElementById("saved-addresses"),d=r&&(r.dataset.jsRender==="true"||r.children.length===0);r&&t.length>0&&(d&&(r.innerHTML=`
                    <div class="mb-4">
                        <label class="block text-sm font-medium text-gray-700 mb-2">Saved Addresses</label>
                        <div class="space-y-2">
                            ${t.map(l=>`
                                <label class="flex items-start p-3 border border-gray-200 rounded-lg cursor-pointer hover:border-primary-500 transition-colors">
                                    <input type="radio" name="saved_address" value="${l.id}" class="mt-1 text-primary-600 focus:ring-primary-500">
                                    <div class="ml-3">
                                        <p class="font-medium text-gray-900">${Templates.escapeHtml(l.full_name||`${l.first_name} ${l.last_name}`)}</p>
                                        <p class="text-sm text-gray-600">${Templates.escapeHtml(l.address_line_1)}</p>
                                        ${l.address_line_2?`<p class="text-sm text-gray-600">${Templates.escapeHtml(l.address_line_2)}</p>`:""}
                                        <p class="text-sm text-gray-600">${Templates.escapeHtml(l.city)}, ${Templates.escapeHtml(l.state||"")} ${Templates.escapeHtml(l.postal_code)}</p>
                                        <p class="text-sm text-gray-600">${Templates.escapeHtml(l.country)}</p>
                                        ${l.is_default?'<span class="inline-block mt-1 px-2 py-0.5 bg-primary-100 text-primary-700 text-xs rounded">Default</span>':""}
                                    </div>
                                </label>
                            `).join("")}
                            <label class="flex items-center p-3 border border-gray-200 rounded-lg cursor-pointer hover:border-primary-500 transition-colors">
                                <input type="radio" name="saved_address" value="new" class="text-primary-600 focus:ring-primary-500" checked>
                                <span class="ml-3 text-gray-700">Enter a new address</span>
                            </label>
                        </div>
                    </div>
                `),c())}catch(x){console.error("Failed to load addresses:",x)}}function c(){let x=document.querySelectorAll('input[name="saved_address"]'),t=document.getElementById("new-address-form");x.forEach(r=>{r.addEventListener("change",d=>{d.target.value==="new"?t?.classList.remove("hidden"):(t?.classList.add("hidden"),e.shipping_address=d.target.value)})})}function u(){let x=document.querySelectorAll("[data-step]"),t=document.querySelectorAll("[data-step-indicator]"),r=document.querySelectorAll("[data-next-step]"),d=document.querySelectorAll("[data-prev-step]");function l(p){x.forEach(f=>{f.classList.toggle("hidden",parseInt(f.dataset.step)!==p)}),t.forEach(f=>{let I=parseInt(f.dataset.stepIndicator);f.classList.toggle("bg-primary-600",I<=p),f.classList.toggle("text-white",I<=p),f.classList.toggle("bg-gray-200",I>p),f.classList.toggle("text-gray-600",I>p)}),s=p}r.forEach(p=>{p.addEventListener("click",async()=>{await m()&&(s===1&&await J(),l(s+1),window.scrollTo({top:0,behavior:"smooth"}))})}),d.forEach(p=>{p.addEventListener("click",()=>{l(s-1),window.scrollTo({top:0,behavior:"smooth"})})}),l(1)}async function m(){switch(s){case 1:return D();case 2:return ye();case 3:return xe();default:return!0}}function b(x){x&&(x.querySelectorAll("[data-error-for]").forEach(t=>t.remove()),x.querySelectorAll('[class*="!border-red-500"]').forEach(t=>t.classList.remove("!border-red-500")))}function v(x,t){if(!x)return;let r=x.getAttribute("name")||x.id||Math.random().toString(36).slice(2,8),d=x.closest("form")?.querySelector(`[data-error-for="${r}"]`);d&&d.remove();let l=document.createElement("p");l.className="text-sm text-red-600 mt-1",l.setAttribute("data-error-for",r),l.textContent=t,x.classList.add("!border-red-500"),x.nextSibling?x.parentNode.insertBefore(l,x.nextSibling):x.parentNode.appendChild(l)}function P(x){if(!x)return;let t=x.querySelector("[data-error-for]");if(!t)return;let r=t.getAttribute("data-error-for"),d=x.querySelector(`[name="${r}"]`)||x.querySelector(`#${r}`)||t.previousElementSibling;if(d&&typeof d.focus=="function")try{d.focus({preventScroll:!0})}catch{d.focus()}}function D(){let x=document.querySelector('input[name="saved_address"]:checked');if(x&&x.value!=="new")return b(document.getElementById("new-address-form")||document.getElementById("information-form")),e.shipping_address=x.value,!0;let t=document.getElementById("shipping-address-form")||document.getElementById("information-form")||document.getElementById("new-address-form");if(!t)return!1;b(t);let r=new FormData(t),d={first_name:r.get("first_name")||r.get("full_name")?.split(" ")?.[0],last_name:r.get("last_name")||(r.get("full_name")?r.get("full_name").split(" ").slice(1).join(" "):""),email:r.get("email"),phone:r.get("phone"),address_line_1:r.get("address_line1")||r.get("address_line_1"),address_line_2:r.get("address_line2")||r.get("address_line_2"),city:r.get("city"),state:r.get("state"),postal_code:r.get("postal_code"),country:r.get("country")},p=["email","first_name","address_line_1","city","postal_code"].filter(f=>!d[f]);if(p.length>0)return p.forEach(f=>{let I=`[name="${f}"]`;f==="address_line_1"&&(I='[name="address_line1"],[name="address_line_1"]');let F=t.querySelector(I);v(F||t,f.replace("_"," ").replace(/\b\w/g,Q=>Q.toUpperCase())+" is required.")}),P(t),!1;if(d.email&&!/^[^@\s]+@[^@\s]+\.[^@\s]+$/.test(d.email)){let f=t.querySelector('[name="email"]');return v(f||t,"Please enter a valid email address."),P(t),!1}return e.shipping_address=d,!0}async function J(){let x=document.getElementById("shipping-methods");if(x){Loader.show(x,"spinner");try{let t=e.shipping_address;if(!t){x.innerHTML='<p class="text-gray-500">Please provide a shipping address to view shipping methods.</p>';return}let r=typeof t=="object"?{country:t.country,postal_code:t.postal_code,city:t.city}:{address_id:t},l=(await ShippingApi.getRates(r)).data||[];if(l.length===0){x.innerHTML='<p class="text-gray-500">No shipping methods available for your location.</p>';return}x.innerHTML=`
                <div class="space-y-3">
                    ${l.map((f,I)=>`
                        <label class="flex items-center justify-between p-4 border border-gray-200 rounded-lg cursor-pointer hover:border-primary-500 transition-colors">
                            <div class="flex items-center">
                                <input 
                                    type="radio" 
                                    name="shipping_method" 
                                    value="${f.id}" 
                                    ${I===0?"checked":""}
                                    class="text-primary-600 focus:ring-primary-500"
                                    data-price="${f.price}"
                                >
                                <div class="ml-3">
                                    <p class="font-medium text-gray-900">${Templates.escapeHtml(f.name)}</p>
                                    ${f.description?`<p class="text-sm text-gray-500">${Templates.escapeHtml(f.description)}</p>`:""}
                                    ${f.estimated_days?`<p class="text-sm text-gray-500">Delivery in ${f.estimated_days} days</p>`:""}
                                </div>
                            </div>
                            <span class="font-semibold text-gray-900">${f.price>0?Templates.formatPrice(f.price):"Free"}</span>
                        </label>
                    `).join("")}
                </div>
            `;let p=x.querySelectorAll('input[name="shipping_method"]');if(p.forEach((f,I)=>{let F=l[I]||{},Q=Number(F.price??F.rate??0)||0,K=F.price_display||F.rate_display||(Q>0?Templates.formatPrice(Q):"Free");f.__price=Q,f.dataset.display=K,F.currency&&F.currency.code&&(f.dataset.currency=F.currency.code),f.addEventListener("change",()=>{we(parseFloat(f.__price)||0,{rateId:f.value,display:f.dataset.display,currency:f.dataset.currency||null,persist:!0,type:"delivery"})})}),l.length>0){e.shipping_method=l[0].id;let f=l[0]||{},I=Number(f.price??f.rate??0)||0,F=f.price_display||f.rate_display||(I>0?Templates.formatPrice(I):"Free");we(I,{rateId:p[0]?.value||f.id,display:F,currency:p[0]?.dataset?.currency||null,persist:!0,type:"delivery"})}}catch(t){console.error("Failed to load shipping methods:",t),x.innerHTML='<p class="text-red-500">Failed to load shipping methods. Please try again.</p>'}}}function le(x){let t=document.getElementById("submit-button"),r=document.getElementById("button-text");!t||!r||(t.disabled=!x,r.textContent=x?"Continue to Review":"No payment methods available")}async function be(){let x=document.getElementById("payment-methods-container");if(x)try{let t=new URLSearchParams;window.CONFIG&&CONFIG.shippingData&&CONFIG.shippingData.countryCode&&t.set("country",CONFIG.shippingData.countryCode),n&&(n.total||n.total===0)&&t.set("amount",n.total);let d=await(await fetch(`/api/v1/payments/gateways/available/?${t.toString()}`,{credentials:"same-origin"})).json(),l=d&&d.data||[],p=x.querySelectorAll(".payment-option");if(p&&p.length>0){try{let I=Array.from(p).map(K=>K.dataset.gateway).filter(Boolean),F=(l||[]).map(K=>K.code);if(I.length===F.length&&I.every((K,ae)=>K===F[ae])){pe(),le(I.length>0);return}}catch(I){console.warn("Failed to compare existing payment gateways:",I)}if(l.length===0){le(p.length>0);return}}if(!l||l.length===0){x.innerHTML=`
                    <div class="text-center py-8 border-2 border-dashed border-gray-300 dark:border-gray-600 rounded-xl">
                        <svg class="w-16 h-16 mx-auto mb-4 text-gray-300 dark:text-gray-600" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M20 13V6a2 2 0 00-2-2H6a2 2 0 00-2 2v7m16 0v5a2 2 0 01-2 2H6a2 2 0 01-2-2v-5m16 0h-2.586a1 1 0 00-.707.293l-2.414 2.414a1 1 0 01-.707.293h-3.172a1 1 0 01-.707-.293l-2.414-2.414A1 1 0 006.586 13H4"></path></svg>
                        <h3 class="text-lg font-medium text-gray-900 dark:text-white mb-2">No payment methods are configured</h3>
                        <p class="text-gray-500 dark:text-gray-400 mb-2">We don't have any payment providers configured for your currency or location. Please contact support to enable online payments.</p>
                        <p class="text-sm text-gray-400">You can still place an order if Cash on Delivery or Bank Transfer is available from admin.</p>
                  </div>
              `,le(!1);return}let f=document.createDocumentFragment();l.forEach((I,F)=>{let Q=document.createElement("div");Q.className="relative payment-option transform transition-all duration-300 hover:scale-[1.01]",Q.dataset.gateway=I.code,Q.style.animation="slideIn 0.3s ease-out both",Q.style.animationDelay=`${F*80}ms`;let K=document.createElement("input");K.type="radio",K.name="payment_method",K.value=I.code,K.id=`payment-${I.code}`,K.className="peer sr-only",K.dataset.feeType=I.fee_type||"none",K.dataset.feeAmount=I.fee_amount_converted??I.fee_amount??0,K.dataset.feePercent=I.fee_amount??0,K.dataset.feeName=I.name||"",F===0&&(K.checked=!0);let ae=document.createElement("label");ae.setAttribute("for",K.id),ae.className="flex items-center justify-between p-4 border-2 rounded-xl cursor-pointer transition-all duration-300 hover:border-gray-400 border-gray-200",ae.innerHTML=`
                    <div class="flex items-center">
                        <div class="w-12 h-12 rounded-full bg-gray-100 flex items-center justify-center mr-4">
                            ${I.icon_url?`<img src="${I.icon_url}" class="h-6" alt="${I.name}">`:`<span class="font-bold">${I.code.toUpperCase()}</span>`}
                        </div>
                        <div>
                            <p class="font-medium text-gray-900 dark:text-white">${Templates.escapeHtml(I.name)}</p>
                            <p class="text-sm text-gray-500 dark:text-gray-400">${Templates.escapeHtml(I.description||"")}</p>
                            ${I.fee_text?`<p class="text-xs text-amber-600 dark:text-amber-400 mt-1">${Templates.escapeHtml(I.fee_text)}</p>`:""}
                            ${I.instructions?`<p class="text-xs text-gray-500 dark:text-gray-400 mt-2">${I.instructions}</p>`:""}
                        </div>
                    </div>
                `,Q.appendChild(K),Q.appendChild(ae);let ge=document.createElement("div");ge.className="absolute top-4 right-4 opacity-0 peer-checked:opacity-100 transition-opacity duration-300",ge.innerHTML='<svg class="w-6 h-6 text-primary-600" fill="currentColor" viewBox="0 0 24 24"><path d="M12 2C6.48 2 2 6.48 2 12s4.48 10 10 10 10-4.48 10-10S17.52 2 12 2zm-2 15l-5-5 1.41-1.41L10 14.17l7.59-7.59L19 8l-9 9z"></path></svg>',Q.appendChild(ge),f.appendChild(Q),I.public_key&&I.requires_client&&ke(I.public_key).catch(Te=>console.error("Failed to load Stripe:",Te))}),x.replaceChildren(f),pe(),le(l.length>0),U()}catch(t){console.error("Failed to load payment gateways:",t)}}function ke(x){return new Promise((t,r)=>{if(window.Stripe&&window.STRIPE_PUBLISHABLE_KEY===x){t();return}if(window.STRIPE_PUBLISHABLE_KEY=x,window.Stripe){fe(x),t();return}let d=document.createElement("script");d.src="https://js.stripe.com/v3/",d.async=!0,d.onload=()=>{try{fe(x),t()}catch(l){r(l)}},d.onerror=l=>r(l),document.head.appendChild(d)})}function fe(x){if(typeof Stripe>"u")throw new Error("Stripe script not loaded");try{let t=Stripe(x),d=t.elements().create("card"),l=document.getElementById("card-element");l&&(l.innerHTML="",d.mount("#card-element"),d.on("change",p=>{let f=document.getElementById("card-errors");f&&(f.textContent=p.error?p.error.message:"")}),window.stripeInstance=t,window.stripeCard=d)}catch(t){throw console.error("Error initializing Stripe elements:",t),t}}(function(){document.addEventListener("DOMContentLoaded",()=>{setTimeout(()=>{be()},50)})})();function we(x,t={}){let r=document.getElementById("shipping-cost"),d=document.getElementById("order-total"),l=w(x,0),p=t.display||(l>0?Templates.formatPrice(l,t.currency||null):"Free");if(r&&(r.textContent=p,r.dataset.price=String(l),t.currency&&(r.dataset.currency=t.currency)),t.persist)try{localStorage.setItem(_,JSON.stringify({type:t.type||"delivery",rateId:t.rateId||null,cost:l,display:p||"",currency:t.currency||null,ts:Date.now()}))}catch{}if(d&&n){let f=(n.total||0)+l;d.textContent=Templates.formatPrice(f),d.dataset.price=f}}function ye(){let x=document.querySelector('input[name="shipping_method"]:checked');return x?(e.shipping_method=x.value,!0):(Toast.error("Please select a shipping method."),!1)}function ce(){let x=document.getElementById("order-summary-toggle"),t=document.getElementById("order-summary-block");if(!x||!t)return;x.addEventListener("click",()=>{let d=t.classList.toggle("hidden");x.setAttribute("aria-expanded",(!d).toString());let l=x.querySelector("svg");l&&l.classList.toggle("rotate-180",!d)});let r=window.getComputedStyle(t).display==="none"||t.classList.contains("hidden");x.setAttribute("aria-expanded",(!r).toString())}function xe(){let x=document.querySelector('input[name="payment_method"]:checked'),t=document.getElementById("payment-form");if(b(t),!x){let d=document.getElementById("payment-methods-container")||t;return v(d,"Please select a payment method."),P(d),!1}let r=x.value;if(r==="stripe"){let d=document.getElementById("cardholder-name");if(!d||!d.value.trim())return v(d||t,"Cardholder name is required."),P(t),!1;if(!window.stripeCard)return v(document.getElementById("card-element")||t,"Card input not ready. Please wait and try again."),!1}if(r==="bkash"){let d=document.getElementById("bkash-number");if(!d||!d.value.trim())return v(d||t,"bKash mobile number is required."),P(t),!1}if(r==="nagad"){let d=document.getElementById("nagad-number");if(!d||!d.value.trim())return v(d||t,"Nagad mobile number is required."),P(t),!1}return e.payment_method=r,!0}function pe(){let x=document.getElementById("same-as-shipping"),t=document.getElementById("billing-address-form");x?.addEventListener("change",p=>{e.same_as_shipping=p.target.checked,t?.classList.toggle("hidden",p.target.checked)}),document.querySelectorAll('input[name="payment_method"]').forEach(p=>{let f=I=>{document.querySelectorAll("[data-payment-form]").forEach(K=>{K.classList.add("hidden")});let F=I.target?I.target.value:p.value||null;if(!F)return;let Q=document.querySelector(`[data-payment-form="${F}"]`);Q||(Q=document.getElementById(`${F}-form`)),Q?.classList.remove("hidden"),U()};p.addEventListener("change",f),p.checked&&f({target:p})});let d=document.getElementById("place-order-btn"),l=document.getElementById("place-order-form");d&&(!l||!l.action||l.action.includes("javascript"))&&d.addEventListener("click",async p=>{p.preventDefault(),await me()})}function Ee(){let x=document.getElementById("information-form");x&&x.dataset.ajaxBound!=="true"&&(x.dataset.ajaxBound="true",x.addEventListener("submit",r=>{r.preventDefault(),r.stopImmediatePropagation(),y(x,{validate:D,redirectUrl:"/checkout/shipping/",loadingText:"Processing..."})},!0));let t=document.getElementById("shipping-form");t&&t.dataset.ajaxBound!=="true"&&(t.dataset.ajaxBound="true",t.addEventListener("submit",r=>{if(r.preventDefault(),r.stopImmediatePropagation(),(document.getElementById("shipping-type")?.value||"delivery")==="pickup"){if(!document.querySelector('input[name="pickup_location"]:checked')){Toast.error("Please select a pickup location.");return}}else if(!(document.querySelector('input[name="shipping_rate_id"]:checked')||document.querySelector('input[name="shipping_method"]:checked'))){Toast.error("Please select a shipping method.");return}y(t,{redirectUrl:"/checkout/payment/",loadingText:"Processing..."})},!0))}async function me(){if(!xe())return;let x=document.getElementById("place-order-btn");x.disabled=!0,x.innerHTML='<svg class="animate-spin h-5 w-5 mx-auto" fill="none" viewBox="0 0 24 24"><circle class="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" stroke-width="4"></circle><path class="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path></svg>';try{let t=document.getElementById("order-notes")?.value;if(e.notes=t||"",!e.same_as_shipping){let l=document.getElementById("billing-address-form");if(l){let p=new FormData(l);e.billing_address={first_name:p.get("billing_first_name"),last_name:p.get("billing_last_name"),address_line_1:p.get("billing_address_line_1"),address_line_2:p.get("billing_address_line_2"),city:p.get("billing_city"),state:p.get("billing_state"),postal_code:p.get("billing_postal_code"),country:p.get("billing_country")}}}let d=(await CheckoutApi.createOrder(e)).data;e.payment_method==="stripe"||e.payment_method==="card"?await Ce(d):e.payment_method==="paypal"?await _e(d):window.location.href=`/orders/${d.id}/confirmation/`}catch(t){console.error("Failed to place order:",t),Toast.error(t.message||"Failed to place order. Please try again."),x.disabled=!1,x.textContent="Place Order"}}async function Ce(x){try{let t=await CheckoutApi.createPaymentIntent(x.id),{client_secret:r}=t.data,d=t.data.publishable_key||window.STRIPE_PUBLISHABLE_KEY||(window.stripeInstance?window.STRIPE_PUBLISHABLE_KEY:null);if(typeof Stripe>"u"&&!window.stripeInstance)throw new Error("Stripe is not loaded.");let p=await(window.stripeInstance||Stripe(d)).confirmCardPayment(r,{payment_method:{card:window.stripeCard,billing_details:{name:`${e.shipping_address.first_name} ${e.shipping_address.last_name}`}}});if(p.error)throw new Error(p.error.message);window.location.href=`/orders/${x.id}/confirmation/`}catch(t){console.error("Stripe payment failed:",t),Toast.error(t.message||"Payment failed. Please try again.");let r=document.getElementById("place-order-btn");r.disabled=!1,r.textContent="Place Order"}}async function _e(x){try{let t=await CheckoutApi.createPayPalOrder(x.id),{approval_url:r}=t.data;window.location.href=r}catch(t){console.error("PayPal payment failed:",t),Toast.error(t.message||"Payment failed. Please try again.");let r=document.getElementById("place-order-btn");r.disabled=!1,r.textContent="Place Order"}}function $e(){n=null,e={shipping_address:null,billing_address:null,same_as_shipping:!0,shipping_method:null,payment_method:null,notes:""},s=1}return{init:B,destroy:$e}})();window.CheckoutPage=Ks;zr=Ks});var sr={};te(sr,{default:()=>Rr});var tr,Rr,rr=ee(()=>{tr=(function(){"use strict";async function n(){L(),await Y(),e()}function e(){s(),o(),h(),w(),_()}function s(){let z=document.getElementById("contact-map");if(!z)return;let E=z.dataset.lat||"0",H=z.dataset.lng||"0",U=z.dataset.address||"Our Location";z.innerHTML=`
            <div class="relative w-full h-64 md:h-80 rounded-2xl overflow-hidden bg-stone-100 dark:bg-stone-800 group">
                <iframe 
                    src="https://www.google.com/maps/embed?pb=!1m18!1m12!1m3!1d3022.9663095343008!2d${H}!3d${E}!2m3!1f0!2f0!3f0!3m2!1i1024!2i768!4f13.1!3m3!1m2!1s0x0%3A0x0!2zM!5e0!3m2!1sen!2sus!4v1234567890"
                    class="w-full h-full border-0"
                    allowfullscreen=""
                    loading="lazy"
                    referrerpolicy="no-referrer-when-downgrade"
                ></iframe>
                <a 
                    href="https://www.google.com/maps/search/?api=1&query=${encodeURIComponent(U)}"
                    target="_blank"
                    rel="noopener noreferrer"
                    class="absolute bottom-4 right-4 px-4 py-2 bg-white dark:bg-stone-800 rounded-xl shadow-lg flex items-center gap-2 text-sm font-medium text-stone-700 dark:text-stone-200 hover:bg-stone-50 dark:hover:bg-stone-700 transition-colors"
                >
                    <svg class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M10 6H6a2 2 0 00-2 2v10a2 2 0 002 2h10a2 2 0 002-2v-4M14 4h6m0 0v6m0-6L10 14"/></svg>
                    Open in Google Maps
                </a>
            </div>
        `}function o(){let z=document.getElementById("live-chat-cta");z&&(z.innerHTML=`
            <div class="bg-gradient-to-br from-primary-600 to-primary-700 dark:from-amber-600 dark:to-amber-700 rounded-2xl p-6 text-white">
                <div class="flex items-start gap-4">
                    <div class="w-12 h-12 bg-white/20 backdrop-blur rounded-xl flex items-center justify-center flex-shrink-0">
                        <svg class="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M8 12h.01M12 12h.01M16 12h.01M21 12c0 4.418-4.03 8-9 8a9.863 9.863 0 01-4.255-.949L3 20l1.395-3.72C3.512 15.042 3 13.574 3 12c0-4.418 4.03-8 9-8s9 3.582 9 8z"/></svg>
                    </div>
                    <div class="flex-1">
                        <h3 class="font-bold text-lg">Need Instant Help?</h3>
                        <p class="text-white/90 text-sm mt-1 mb-4">Our support team is online and ready to assist you right now.</p>
                        <button id="open-live-chat" class="inline-flex items-center gap-2 px-4 py-2 bg-white text-primary-700 dark:text-amber-700 font-semibold rounded-xl hover:bg-white/90 transition-colors">
                            <span class="relative flex h-2 w-2">
                                <span class="animate-ping absolute inline-flex h-full w-full rounded-full bg-green-400 opacity-75"></span>
                                <span class="relative inline-flex rounded-full h-2 w-2 bg-green-500"></span>
                            </span>
                            Start Live Chat
                        </button>
                    </div>
                </div>
            </div>
        `,document.getElementById("open-live-chat")?.addEventListener("click",()=>{document.dispatchEvent(new CustomEvent("chat:open"))}))}function h(){let z=document.getElementById("quick-contact");if(!z)return;let E=[{icon:"phone",label:"Call Us",action:"tel:",color:"emerald"},{icon:"whatsapp",label:"WhatsApp",action:"https://wa.me/",color:"green"},{icon:"email",label:"Email",action:"mailto:",color:"blue"}];z.innerHTML=`
            <div class="grid grid-cols-3 gap-4">
                <a href="tel:+1234567890" class="flex flex-col items-center gap-2 p-4 bg-white dark:bg-stone-800 rounded-xl border border-stone-200 dark:border-stone-700 hover:border-emerald-500 dark:hover:border-emerald-500 hover:shadow-lg transition-all group">
                    <div class="w-12 h-12 bg-emerald-100 dark:bg-emerald-900/30 rounded-xl flex items-center justify-center text-emerald-600 dark:text-emerald-400 group-hover:scale-110 transition-transform">
                        <svg class="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M3 5a2 2 0 012-2h3.28a1 1 0 01.948.684l1.498 4.493a1 1 0 01-.502 1.21l-2.257 1.13a11.042 11.042 0 005.516 5.516l1.13-2.257a1 1 0 011.21-.502l4.493 1.498a1 1 0 01.684.949V19a2 2 0 01-2 2h-1C9.716 21 3 14.284 3 6V5z"/></svg>
                    </div>
                    <span class="text-sm font-medium text-stone-700 dark:text-stone-300">Call Us</span>
                </a>
                <a href="https://wa.me/1234567890" target="_blank" rel="noopener noreferrer" class="flex flex-col items-center gap-2 p-4 bg-white dark:bg-stone-800 rounded-xl border border-stone-200 dark:border-stone-700 hover:border-green-500 dark:hover:border-green-500 hover:shadow-lg transition-all group">
                    <div class="w-12 h-12 bg-green-100 dark:bg-green-900/30 rounded-xl flex items-center justify-center text-green-600 dark:text-green-400 group-hover:scale-110 transition-transform">
                        <svg class="w-6 h-6" fill="currentColor" viewBox="0 0 24 24"><path d="M17.472 14.382c-.297-.149-1.758-.867-2.03-.967-.273-.099-.471-.148-.67.15-.197.297-.767.966-.94 1.164-.173.199-.347.223-.644.075-.297-.15-1.255-.463-2.39-1.475-.883-.788-1.48-1.761-1.653-2.059-.173-.297-.018-.458.13-.606.134-.133.298-.347.446-.52.149-.174.198-.298.298-.497.099-.198.05-.371-.025-.52-.075-.149-.669-1.612-.916-2.207-.242-.579-.487-.5-.669-.51-.173-.008-.371-.01-.57-.01-.198 0-.52.074-.792.372-.272.297-1.04 1.016-1.04 2.479 0 1.462 1.065 2.875 1.213 3.074.149.198 2.096 3.2 5.077 4.487.709.306 1.262.489 1.694.625.712.227 1.36.195 1.871.118.571-.085 1.758-.719 2.006-1.413.248-.694.248-1.289.173-1.413-.074-.124-.272-.198-.57-.347m-5.421 7.403h-.004a9.87 9.87 0 01-5.031-1.378l-.361-.214-3.741.982.998-3.648-.235-.374a9.86 9.86 0 01-1.51-5.26c.001-5.45 4.436-9.884 9.888-9.884 2.64 0 5.122 1.03 6.988 2.898a9.825 9.825 0 012.893 6.994c-.003 5.45-4.437 9.884-9.885 9.884m8.413-18.297A11.815 11.815 0 0012.05 0C5.495 0 .16 5.335.157 11.892c0 2.096.547 4.142 1.588 5.945L.057 24l6.305-1.654a11.882 11.882 0 005.683 1.448h.005c6.554 0 11.89-5.335 11.893-11.893a11.821 11.821 0 00-3.48-8.413z"/></svg>
                    </div>
                    <span class="text-sm font-medium text-stone-700 dark:text-stone-300">WhatsApp</span>
                </a>
                <a href="mailto:support@bunoraa.com" class="flex flex-col items-center gap-2 p-4 bg-white dark:bg-stone-800 rounded-xl border border-stone-200 dark:border-stone-700 hover:border-blue-500 dark:hover:border-blue-500 hover:shadow-lg transition-all group">
                    <div class="w-12 h-12 bg-blue-100 dark:bg-blue-900/30 rounded-xl flex items-center justify-center text-blue-600 dark:text-blue-400 group-hover:scale-110 transition-transform">
                        <svg class="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M3 8l7.89 5.26a2 2 0 002.22 0L21 8M5 19h14a2 2 0 002-2V7a2 2 0 00-2-2H5a2 2 0 00-2 2v10a2 2 0 002 2z"/></svg>
                    </div>
                    <span class="text-sm font-medium text-stone-700 dark:text-stone-300">Email</span>
                </a>
            </div>
        `}function w(){let z=document.getElementById("faq-preview");if(!z)return;let E=[{q:"How long does shipping take?",a:"Standard shipping takes 5-7 business days. Express options are available at checkout."},{q:"What is your return policy?",a:"We offer a 30-day hassle-free return policy on all unused items in original packaging."},{q:"Do you ship internationally?",a:"Yes! We ship to over 100 countries worldwide."}];z.innerHTML=`
            <div class="bg-white dark:bg-stone-800 rounded-2xl border border-stone-200 dark:border-stone-700 overflow-hidden">
                <div class="px-6 py-4 border-b border-stone-200 dark:border-stone-700 flex items-center justify-between">
                    <h3 class="font-semibold text-stone-900 dark:text-white">Frequently Asked Questions</h3>
                    <a href="/faq/" class="text-sm text-primary-600 dark:text-amber-400 hover:underline">View All</a>
                </div>
                <div class="divide-y divide-stone-200 dark:divide-stone-700">
                    ${E.map((H,U)=>`
                        <div class="faq-item" data-index="${U}">
                            <button class="faq-trigger w-full px-6 py-4 flex items-center justify-between text-left hover:bg-stone-50 dark:hover:bg-stone-700/50 transition-colors">
                                <span class="font-medium text-stone-900 dark:text-white pr-4">${Templates.escapeHtml(H.q)}</span>
                                <svg class="faq-icon w-5 h-5 text-stone-400 transform transition-transform flex-shrink-0" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M19 9l-7 7-7-7"/></svg>
                            </button>
                            <div class="faq-answer hidden px-6 pb-4 text-stone-600 dark:text-stone-400">
                                ${Templates.escapeHtml(H.a)}
                            </div>
                        </div>
                    `).join("")}
                </div>
            </div>
        `,z.querySelectorAll(".faq-trigger").forEach(H=>{H.addEventListener("click",()=>{let U=H.closest(".faq-item"),B=U.querySelector(".faq-answer"),C=U.querySelector(".faq-icon"),k=!B.classList.contains("hidden");z.querySelectorAll(".faq-item").forEach(S=>{S!==U&&(S.querySelector(".faq-answer").classList.add("hidden"),S.querySelector(".faq-icon").classList.remove("rotate-180"))}),B.classList.toggle("hidden"),C.classList.toggle("rotate-180")})})}function _(){let z=document.getElementById("office-status");if(!z)return;let E={start:9,end:18,timezone:"America/New_York",days:[1,2,3,4,5]};function H(){let U=new Date,B=U.getDay(),C=U.getHours(),S=E.days.includes(B)&&C>=E.start&&C<E.end;z.innerHTML=`
                <div class="flex items-center gap-3 p-4 rounded-xl ${S?"bg-emerald-50 dark:bg-emerald-900/20 border border-emerald-200 dark:border-emerald-800":"bg-stone-100 dark:bg-stone-800 border border-stone-200 dark:border-stone-700"}">
                    <span class="relative flex h-3 w-3">
                        ${S?`<span class="animate-ping absolute inline-flex h-full w-full rounded-full bg-emerald-400 opacity-75"></span>
                             <span class="relative inline-flex rounded-full h-3 w-3 bg-emerald-500"></span>`:'<span class="relative inline-flex rounded-full h-3 w-3 bg-stone-400"></span>'}
                    </span>
                    <div>
                        <p class="font-medium ${S?"text-emerald-700 dark:text-emerald-400":"text-stone-600 dark:text-stone-400"}">
                            ${S?"We're Open!":"Currently Closed"}
                        </p>
                        <p class="text-xs ${S?"text-emerald-600 dark:text-emerald-500":"text-stone-500 dark:text-stone-500"}">
                            ${S?"Our team is available to help you.":`Office hours: Mon-Fri ${E.start}AM - ${E.end>12?E.end-12+"PM":E.end+"AM"}`}
                        </p>
                    </div>
                </div>
            `}H(),setInterval(H,6e4)}function L(){let z=document.getElementById("contact-form");if(!z)return;let E=FormValidator.create(z,{name:{required:!0,minLength:2,maxLength:100},email:{required:!0,email:!0},subject:{required:!0,minLength:5,maxLength:200},message:{required:!0,minLength:20,maxLength:2e3}});z.addEventListener("submit",async H=>{if(H.preventDefault(),!E.validate()){Toast.error("Please fill in all required fields correctly.");return}let U=z.querySelector('button[type="submit"]'),B=U.textContent;U.disabled=!0,U.innerHTML='<svg class="animate-spin h-5 w-5 mx-auto" fill="none" viewBox="0 0 24 24"><circle class="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" stroke-width="4"></circle><path class="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path></svg>';try{let C=new FormData(z),k={name:C.get("name"),email:C.get("email"),phone:C.get("phone"),subject:C.get("subject"),message:C.get("message")};await SupportApi.submitContactForm(k),Toast.success("Thank you for your message! We'll get back to you soon."),z.reset(),E.clearErrors(),A()}catch(C){Toast.error(C.message||"Failed to send message. Please try again.")}finally{U.disabled=!1,U.textContent=B}})}function A(){let z=document.getElementById("contact-form"),E=document.getElementById("contact-success");z&&E&&(z.classList.add("hidden"),E.classList.remove("hidden"),E.innerHTML=`
                <div class="text-center py-12">
                    <div class="w-16 h-16 bg-green-100 rounded-full flex items-center justify-center mx-auto mb-4">
                        <svg class="w-8 h-8 text-green-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M5 13l4 4L19 7"/>
                        </svg>
                    </div>
                    <h3 class="text-xl font-semibold text-gray-900 mb-2">Message Sent!</h3>
                    <p class="text-gray-600 mb-6">Thank you for reaching out. We'll respond to your inquiry within 24-48 hours.</p>
                    <button id="send-another-btn" class="px-6 py-2 bg-primary-600 text-white rounded-lg hover:bg-primary-700 transition-colors">
                        Send Another Message
                    </button>
                </div>
            `,document.getElementById("send-another-btn")?.addEventListener("click",()=>{z.classList.remove("hidden"),E.classList.add("hidden")}))}async function Y(){let z=document.getElementById("contact-info");if(z)try{let H=(await PagesApi.getContactInfo()).data;if(!H)return;z.innerHTML=`
                <div class="space-y-6">
                    ${H.address?`
                        <div class="flex gap-4">
                            <div class="flex-shrink-0 w-12 h-12 bg-primary-100 rounded-lg flex items-center justify-center">
                                <svg class="w-6 h-6 text-primary-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                    <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M17.657 16.657L13.414 20.9a1.998 1.998 0 01-2.827 0l-4.244-4.243a8 8 0 1111.314 0z"/>
                                    <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M15 11a3 3 0 11-6 0 3 3 0 016 0z"/>
                                </svg>
                            </div>
                            <div>
                                <h3 class="font-semibold text-gray-900">Address</h3>
                                <p class="text-gray-600">${Templates.escapeHtml(H.address)}</p>
                            </div>
                        </div>
                    `:""}
                    
                    ${H.phone?`
                        <div class="flex gap-4">
                            <div class="flex-shrink-0 w-12 h-12 bg-primary-100 rounded-lg flex items-center justify-center">
                                <svg class="w-6 h-6 text-primary-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                    <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M3 5a2 2 0 012-2h3.28a1 1 0 01.948.684l1.498 4.493a1 1 0 01-.502 1.21l-2.257 1.13a11.042 11.042 0 005.516 5.516l1.13-2.257a1 1 0 011.21-.502l4.493 1.498a1 1 0 01.684.949V19a2 2 0 01-2 2h-1C9.716 21 3 14.284 3 6V5z"/>
                                </svg>
                            </div>
                            <div>
                                <h3 class="font-semibold text-gray-900">Phone</h3>
                                <a href="tel:${H.phone}" class="text-gray-600 hover:text-primary-600">${Templates.escapeHtml(H.phone)}</a>
                            </div>
                        </div>
                    `:""}
                    
                    ${H.email?`
                        <div class="flex gap-4">
                            <div class="flex-shrink-0 w-12 h-12 bg-primary-100 rounded-lg flex items-center justify-center">
                                <svg class="w-6 h-6 text-primary-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                    <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M3 8l7.89 5.26a2 2 0 002.22 0L21 8M5 19h14a2 2 0 002-2V7a2 2 0 00-2-2H5a2 2 0 00-2 2v10a2 2 0 002 2z"/>
                                </svg>
                            </div>
                            <div>
                                <h3 class="font-semibold text-gray-900">Email</h3>
                                <a href="mailto:${H.email}" class="text-gray-600 hover:text-primary-600">${Templates.escapeHtml(H.email)}</a>
                            </div>
                        </div>
                    `:""}
                    
                    ${H.business_hours?`
                        <div class="flex gap-4">
                            <div class="flex-shrink-0 w-12 h-12 bg-primary-100 rounded-lg flex items-center justify-center">
                                <svg class="w-6 h-6 text-primary-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                    <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 8v4l3 3m6-3a9 9 0 11-18 0 9 9 0 0118 0z"/>
                                </svg>
                            </div>
                            <div>
                                <h3 class="font-semibold text-gray-900">Business Hours</h3>
                                <p class="text-gray-600">${Templates.escapeHtml(H.business_hours)}</p>
                            </div>
                        </div>
                    `:""}
                    
                    ${H.social_links&&Object.keys(H.social_links).length>0?`
                        <div class="pt-4 border-t border-gray-200">
                            <h3 class="font-semibold text-gray-900 mb-3">Follow Us</h3>
                            <div class="flex gap-3">
                                ${H.social_links.facebook?`
                                    <a href="${H.social_links.facebook}" target="_blank" class="w-10 h-10 bg-gray-100 rounded-lg flex items-center justify-center hover:bg-gray-200 transition-colors">
                                        <svg class="w-5 h-5 text-[#1877F2]" fill="currentColor" viewBox="0 0 24 24"><path d="M24 12.073c0-6.627-5.373-12-12-12s-12 5.373-12 12c0 5.99 4.388 10.954 10.125 11.854v-8.385H7.078v-3.47h3.047V9.43c0-3.007 1.792-4.669 4.533-4.669 1.312 0 2.686.235 2.686.235v2.953H15.83c-1.491 0-1.956.925-1.956 1.874v2.25h3.328l-.532 3.47h-2.796v8.385C19.612 23.027 24 18.062 24 12.073z"/></svg>
                                    </a>
                                `:""}
                                ${H.social_links.instagram?`
                                    <a href="${H.social_links.instagram}" target="_blank" class="w-10 h-10 bg-gray-100 rounded-lg flex items-center justify-center hover:bg-gray-200 transition-colors">
                                        <svg class="w-5 h-5 text-[#E4405F]" fill="currentColor" viewBox="0 0 24 24"><path d="M12 2.163c3.204 0 3.584.012 4.85.07 3.252.148 4.771 1.691 4.919 4.919.058 1.265.069 1.645.069 4.849 0 3.205-.012 3.584-.069 4.849-.149 3.225-1.664 4.771-4.919 4.919-1.266.058-1.644.07-4.85.07-3.204 0-3.584-.012-4.849-.07-3.26-.149-4.771-1.699-4.919-4.92-.058-1.265-.07-1.644-.07-4.849 0-3.204.013-3.583.07-4.849.149-3.227 1.664-4.771 4.919-4.919 1.266-.057 1.645-.069 4.849-.069zm0-2.163c-3.259 0-3.667.014-4.947.072-4.358.2-6.78 2.618-6.98 6.98-.059 1.281-.073 1.689-.073 4.948 0 3.259.014 3.668.072 4.948.2 4.358 2.618 6.78 6.98 6.98 1.281.058 1.689.072 4.948.072 3.259 0 3.668-.014 4.948-.072 4.354-.2 6.782-2.618 6.979-6.98.059-1.28.073-1.689.073-4.948 0-3.259-.014-3.667-.072-4.947-.196-4.354-2.617-6.78-6.979-6.98-1.281-.059-1.69-.073-4.949-.073zm0 5.838c-3.403 0-6.162 2.759-6.162 6.162s2.759 6.163 6.162 6.163 6.162-2.759 6.162-6.163c0-3.403-2.759-6.162-6.162-6.162zm0 10.162c-2.209 0-4-1.79-4-4 0-2.209 1.791-4 4-4s4 1.791 4 4c0 2.21-1.791 4-4 4zm6.406-11.845c-.796 0-1.441.645-1.441 1.44s.645 1.44 1.441 1.44c.795 0 1.439-.645 1.439-1.44s-.644-1.44-1.439-1.44z"/></svg>
                                    </a>
                                `:""}
                                ${H.social_links.twitter?`
                                    <a href="${H.social_links.twitter}" target="_blank" class="w-10 h-10 bg-gray-100 rounded-lg flex items-center justify-center hover:bg-gray-200 transition-colors">
                                        <svg class="w-5 h-5 text-[#1DA1F2]" fill="currentColor" viewBox="0 0 24 24"><path d="M23.953 4.57a10 10 0 01-2.825.775 4.958 4.958 0 002.163-2.723c-.951.555-2.005.959-3.127 1.184a4.92 4.92 0 00-8.384 4.482C7.69 8.095 4.067 6.13 1.64 3.162a4.822 4.822 0 00-.666 2.475c0 1.71.87 3.213 2.188 4.096a4.904 4.904 0 01-2.228-.616v.06a4.923 4.923 0 003.946 4.827 4.996 4.996 0 01-2.212.085 4.936 4.936 0 004.604 3.417 9.867 9.867 0 01-6.102 2.105c-.39 0-.779-.023-1.17-.067a13.995 13.995 0 007.557 2.209c9.053 0 13.998-7.496 13.998-13.985 0-.21 0-.42-.015-.63A9.935 9.935 0 0024 4.59z"/></svg>
                                    </a>
                                `:""}
                                ${H.social_links.youtube?`
                                    <a href="${H.social_links.youtube}" target="_blank" class="w-10 h-10 bg-gray-100 rounded-lg flex items-center justify-center hover:bg-gray-200 transition-colors">
                                        <svg class="w-5 h-5 text-[#FF0000]" fill="currentColor" viewBox="0 0 24 24"><path d="M23.498 6.186a3.016 3.016 0 0 0-2.122-2.136C19.505 3.545 12 3.545 12 3.545s-7.505 0-9.377.505A3.017 3.017 0 0 0 .502 6.186C0 8.07 0 12 0 12s0 3.93.502 5.814a3.016 3.016 0 0 0 2.122 2.136c1.871.505 9.376.505 9.376.505s7.505 0 9.377-.505a3.015 3.015 0 0 0 2.122-2.136C24 15.93 24 12 24 12s0-3.93-.502-5.814zM9.545 15.568V8.432L15.818 12l-6.273 3.568z"/></svg>
                                    </a>
                                `:""}
                            </div>
                        </div>
                    `:""}
                </div>
            `}catch(E){console.error("Failed to load contact info:",E)}}function Z(){}return{init:n,destroy:Z}})();window.ContactPage=tr;Rr=tr});var nr={};te(nr,{default:()=>Or});var ar,Or,or=ee(()=>{ar=(function(){"use strict";let n=[],e=[];async function s(){let i=document.getElementById("faq-list");i&&i.querySelector(".faq-item")?z():await E(),S(),o()}function o(){h(),w(),_(),L(),A(),Y(),Z()}function h(){if(!document.querySelector(".faq-search-container")||!("webkitSpeechRecognition"in window||"SpeechRecognition"in window))return;let a=window.SpeechRecognition||window.webkitSpeechRecognition,c=new a;c.continuous=!1,c.lang="en-US";let u=document.createElement("button");u.id="faq-voice-search",u.type="button",u.className="absolute right-12 top-1/2 -translate-y-1/2 p-2 text-stone-400 hover:text-primary-600 dark:hover:text-amber-400 transition-colors",u.innerHTML=`
            <svg class="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M19 11a7 7 0 01-7 7m0 0a7 7 0 01-7-7m7 7v4m0 0H8m4 0h4m-4-8a3 3 0 01-3-3V5a3 3 0 116 0v6a3 3 0 01-3 3z"/></svg>
        `;let m=document.getElementById("faq-search");m&&m.parentElement&&(m.parentElement.style.position="relative",m.parentElement.appendChild(u));let b=!1;u.addEventListener("click",()=>{b?c.stop():(c.start(),u.classList.add("text-red-500","animate-pulse")),b=!b}),c.onresult=v=>{let P=v.results[0][0].transcript;m&&(m.value=P,m.dispatchEvent(new Event("input"))),u.classList.remove("text-red-500","animate-pulse"),b=!1},c.onerror=()=>{u.classList.remove("text-red-500","animate-pulse"),b=!1}}function w(){document.querySelectorAll(".faq-content, .accordion-content, .faq-answer").forEach(i=>{if(i.querySelector(".faq-rating"))return;let c=(i.closest(".faq-item")||i.closest("[data-accordion]"))?.dataset.id||Math.random().toString(36).substr(2,9),u=`
                <div class="faq-rating mt-4 pt-4 border-t border-stone-200 dark:border-stone-700 flex items-center justify-between">
                    <span class="text-sm text-stone-500 dark:text-stone-400">Was this answer helpful?</span>
                    <div class="flex gap-2">
                        <button class="faq-rate-btn px-3 py-1 text-sm border border-stone-200 dark:border-stone-600 rounded-lg hover:bg-emerald-50 dark:hover:bg-emerald-900/20 hover:border-emerald-500 hover:text-emerald-600 dark:hover:text-emerald-400 transition-all" data-helpful="yes" data-question="${c}">
                            <span class="flex items-center gap-1">
                                <svg class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M14 10h4.764a2 2 0 011.789 2.894l-3.5 7A2 2 0 0115.263 21h-4.017c-.163 0-.326-.02-.485-.06L7 20m7-10V5a2 2 0 00-2-2h-.095c-.5 0-.905.405-.905.905 0 .714-.211 1.412-.608 2.006L7 11v9m7-10h-2M7 20H5a2 2 0 01-2-2v-6a2 2 0 012-2h2.5"/></svg>
                                Yes
                            </span>
                        </button>
                        <button class="faq-rate-btn px-3 py-1 text-sm border border-stone-200 dark:border-stone-600 rounded-lg hover:bg-red-50 dark:hover:bg-red-900/20 hover:border-red-500 hover:text-red-600 dark:hover:text-red-400 transition-all" data-helpful="no" data-question="${c}">
                            <span class="flex items-center gap-1">
                                <svg class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M10 14H5.236a2 2 0 01-1.789-2.894l3.5-7A2 2 0 018.736 3h4.018a2 2 0 01.485.06l3.76.94m-7 10v5a2 2 0 002 2h.096c.5 0 .905-.405.905-.904 0-.715.211-1.413.608-2.008L17 13V4m-7 10h2m5-10h2a2 2 0 012 2v6a2 2 0 01-2 2h-2.5"/></svg>
                                No
                            </span>
                        </button>
                    </div>
                </div>
            `;i.insertAdjacentHTML("beforeend",u)}),document.addEventListener("click",i=>{let a=i.target.closest(".faq-rate-btn");if(!a)return;let c=a.dataset.helpful==="yes",u=a.dataset.question,m=a.closest(".faq-rating"),b=JSON.parse(localStorage.getItem("faqRatings")||"{}");b[u]=c,localStorage.setItem("faqRatings",JSON.stringify(b)),m.innerHTML=`
                <div class="flex items-center gap-2 text-sm ${c?"text-emerald-600 dark:text-emerald-400":"text-stone-500 dark:text-stone-400"}">
                    <svg class="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z"/></svg>
                    <span>Thanks for your feedback!</span>
                </div>
            `,typeof analytics<"u"&&analytics.track("faq_rated",{questionId:u,helpful:c})})}function _(){let i=document.getElementById("faq-contact-promo");i&&(i.innerHTML=`
            <div class="bg-gradient-to-br from-stone-900 to-stone-800 dark:from-stone-800 dark:to-stone-900 text-white rounded-2xl p-6 md:p-8">
                <div class="flex flex-col md:flex-row items-center gap-6">
                    <div class="w-16 h-16 bg-primary-600 dark:bg-amber-600 rounded-2xl flex items-center justify-center flex-shrink-0">
                        <svg class="w-8 h-8" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M8.228 9c.549-1.165 2.03-2 3.772-2 2.21 0 4 1.343 4 3 0 1.4-1.278 2.575-3.006 2.907-.542.104-.994.54-.994 1.093m0 3h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z"/></svg>
                    </div>
                    <div class="text-center md:text-left flex-1">
                        <h3 class="text-xl font-bold mb-2">Can't Find What You're Looking For?</h3>
                        <p class="text-stone-300 mb-4">Our support team is here to help. Get personalized assistance for your questions.</p>
                        <div class="flex flex-col sm:flex-row gap-3 justify-center md:justify-start">
                            <a href="/contact/" class="inline-flex items-center justify-center gap-2 px-6 py-3 bg-white text-stone-900 font-semibold rounded-xl hover:bg-stone-100 transition-colors">
                                <svg class="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M3 8l7.89 5.26a2 2 0 002.22 0L21 8M5 19h14a2 2 0 002-2V7a2 2 0 00-2-2H5a2 2 0 00-2 2v10a2 2 0 002 2z"/></svg>
                                Contact Support
                            </a>
                            <button id="open-chat-faq" class="inline-flex items-center justify-center gap-2 px-6 py-3 bg-primary-600 dark:bg-amber-600 text-white font-semibold rounded-xl hover:bg-primary-700 dark:hover:bg-amber-700 transition-colors">
                                <span class="relative flex h-2 w-2">
                                    <span class="animate-ping absolute inline-flex h-full w-full rounded-full bg-white opacity-75"></span>
                                    <span class="relative inline-flex rounded-full h-2 w-2 bg-white"></span>
                                </span>
                                Live Chat
                            </button>
                        </div>
                    </div>
                </div>
            </div>
        `,document.getElementById("open-chat-faq")?.addEventListener("click",()=>{document.dispatchEvent(new CustomEvent("chat:open"))}))}function L(){let i=document.getElementById("popular-questions");if(!i)return;let a=JSON.parse(localStorage.getItem("faqRatings")||"{}"),c=Object.entries(a).filter(([m,b])=>b).slice(0,5).map(([m])=>m),u=[];document.querySelectorAll(".faq-item, [data-accordion]").forEach(m=>{let b=m.dataset.id;if(c.includes(b)||u.length<3){let v=m.querySelector("button span, .accordion-toggle span")?.textContent?.trim();v&&u.push({id:b,question:v,element:m})}}),u.length!==0&&(i.innerHTML=`
            <div class="bg-primary-50 dark:bg-amber-900/20 rounded-2xl p-6">
                <h3 class="font-semibold text-stone-900 dark:text-white mb-4 flex items-center gap-2">
                    <svg class="w-5 h-5 text-primary-600 dark:text-amber-400" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M13 7h8m0 0v8m0-8l-8 8-4-4-6 6"/></svg>
                    Most Helpful Questions
                </h3>
                <ul class="space-y-2">
                    ${u.slice(0,5).map(m=>`
                        <li>
                            <button class="popular-q-link text-left text-primary-600 dark:text-amber-400 hover:underline text-sm" data-target="${m.id}">
                                ${Templates.escapeHtml(m.question)}
                            </button>
                        </li>
                    `).join("")}
                </ul>
            </div>
        `,i.querySelectorAll(".popular-q-link").forEach(m=>{m.addEventListener("click",()=>{let b=m.dataset.target,v=document.querySelector(`[data-id="${b}"], .faq-item`);if(v){v.scrollIntoView({behavior:"smooth",block:"center"});let P=v.querySelector(".faq-trigger, .accordion-toggle");P&&P.click()}})}))}function A(){document.querySelectorAll(".faq-content, .accordion-content, .faq-answer").forEach(i=>{if(i.querySelector(".faq-share"))return;let c=(i.closest(".faq-item")||i.closest("[data-accordion]"))?.querySelector("button span, .accordion-toggle span")?.textContent?.trim();if(!c)return;let u=`
                <div class="faq-share flex items-center gap-2 mt-3">
                    <span class="text-xs text-stone-400 dark:text-stone-500">Share:</span>
                    <button class="faq-share-btn p-1.5 hover:bg-stone-100 dark:hover:bg-stone-700 rounded transition-colors" data-platform="copy" title="Copy link">
                        <svg class="w-4 h-4 text-stone-400" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M8 16H6a2 2 0 01-2-2V6a2 2 0 012-2h8a2 2 0 012 2v2m-6 12h8a2 2 0 002-2v-8a2 2 0 00-2-2h-8a2 2 0 00-2 2v8a2 2 0 002 2z"/></svg>
                    </button>
                    <a href="https://twitter.com/intent/tweet?text=${encodeURIComponent("Q: "+c)}&url=${encodeURIComponent(window.location.href)}" target="_blank" rel="noopener noreferrer" class="faq-share-btn p-1.5 hover:bg-stone-100 dark:hover:bg-stone-700 rounded transition-colors" title="Share on Twitter">
                        <svg class="w-4 h-4 text-stone-400" fill="currentColor" viewBox="0 0 24 24"><path d="M18.244 2.25h3.308l-7.227 8.26 8.502 11.24H16.17l-5.214-6.817L4.99 21.75H1.68l7.73-8.835L1.254 2.25H8.08l4.713 6.231zm-1.161 17.52h1.833L7.084 4.126H5.117z"/></svg>
                    </a>
                    <a href="mailto:?subject=${encodeURIComponent("FAQ: "+c)}&body=${encodeURIComponent(window.location.href)}" class="faq-share-btn p-1.5 hover:bg-stone-100 dark:hover:bg-stone-700 rounded transition-colors" title="Share via email">
                        <svg class="w-4 h-4 text-stone-400" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M3 8l7.89 5.26a2 2 0 002.22 0L21 8M5 19h14a2 2 0 002-2V7a2 2 0 00-2-2H5a2 2 0 00-2 2v10a2 2 0 002 2z"/></svg>
                    </a>
                </div>
            `;i.insertAdjacentHTML("beforeend",u)}),document.addEventListener("click",i=>{let a=i.target.closest('.faq-share-btn[data-platform="copy"]');a&&navigator.clipboard.writeText(window.location.href).then(()=>{let c=a.innerHTML;a.innerHTML='<svg class="w-4 h-4 text-emerald-500" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M5 13l4 4L19 7"/></svg>',setTimeout(()=>{a.innerHTML=c},2e3)})})}function Y(){let i=document.querySelectorAll(".faq-item, [data-accordion]"),a=-1;document.addEventListener("keydown",c=>{if(document.querySelector("#faq-container, #faq-list")){if(c.key==="ArrowDown"||c.key==="ArrowUp"){c.preventDefault(),c.key==="ArrowDown"?a=Math.min(a+1,i.length-1):a=Math.max(a-1,0);let u=i[a];if(u){u.scrollIntoView({behavior:"smooth",block:"center"});let m=u.querySelector(".faq-trigger, .accordion-toggle, button");m&&m.focus()}}if(c.key==="Enter"&&a>=0){let m=i[a]?.querySelector(".faq-trigger, .accordion-toggle");m&&m.click()}c.key==="/"&&document.activeElement?.tagName!=="INPUT"&&(c.preventDefault(),document.getElementById("faq-search")?.focus())}})}function Z(){let i=new IntersectionObserver(a=>{a.forEach(c=>{if(c.isIntersecting){let u=c.target,m=u.querySelector("button span, .accordion-toggle span")?.textContent?.trim(),b=JSON.parse(localStorage.getItem("faqViews")||"{}"),v=u.dataset.id||m?.substring(0,30);v&&(b[v]=(b[v]||0)+1,localStorage.setItem("faqViews",JSON.stringify(b)))}})},{threshold:.5});document.querySelectorAll(".faq-item, [data-accordion]").forEach(a=>{i.observe(a)}),document.addEventListener("click",a=>{let c=a.target.closest(".faq-trigger, .accordion-toggle");if(c){let u=c.closest(".faq-item, [data-accordion]"),m=c.querySelector("span")?.textContent?.trim();typeof analytics<"u"&&analytics.track("faq_opened",{question:m?.substring(0,100)})}})}function z(){let i=document.querySelectorAll(".category-tab"),a=document.querySelectorAll(".faq-category");i.forEach(u=>{u.addEventListener("click",m=>{m.preventDefault(),i.forEach(P=>{P.classList.remove("bg-primary-600","dark:bg-amber-600","text-white"),P.classList.add("bg-stone-100","dark:bg-stone-800","text-stone-700","dark:text-stone-300")}),u.classList.add("bg-primary-600","dark:bg-amber-600","text-white"),u.classList.remove("bg-stone-100","dark:bg-stone-800","text-stone-700","dark:text-stone-300");let b=u.dataset.category;b==="all"?a.forEach(P=>P.classList.remove("hidden")):a.forEach(P=>{P.classList.toggle("hidden",P.dataset.category!==b)});let v=document.getElementById("faq-search");v&&(v.value=""),document.querySelectorAll(".faq-item").forEach(P=>P.classList.remove("hidden"))})}),document.querySelectorAll(".accordion-toggle").forEach(u=>{u.addEventListener("click",()=>{let m=u.closest("[data-accordion]"),b=m.querySelector(".accordion-content"),v=m.querySelector(".accordion-icon"),P=!b.classList.contains("hidden");document.querySelectorAll("[data-accordion]").forEach(D=>{D!==m&&(D.querySelector(".accordion-content")?.classList.add("hidden"),D.querySelector(".accordion-icon")?.classList.remove("rotate-180"))}),P?(b.classList.add("hidden"),v.classList.remove("rotate-180")):(b.classList.remove("hidden"),v.classList.add("rotate-180"))})})}async function E(){let i=document.getElementById("faq-container");if(i){Loader.show(i,"skeleton");try{let c=(await PagesApi.getFAQs()).data||[];if(e=c,c.length===0){i.innerHTML=`
                    <div class="text-center py-12">
                        <svg class="w-16 h-16 text-stone-300 dark:text-stone-600 mx-auto mb-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M8.228 9c.549-1.165 2.03-2 3.772-2 2.21 0 4 1.343 4 3 0 1.4-1.278 2.575-3.006 2.907-.542.104-.994.54-.994 1.093m0 3h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z"/>
                        </svg>
                        <p class="text-stone-500 dark:text-stone-400">No FAQs available at the moment.</p>
                    </div>
                `;return}n=H(c),U(n)}catch(a){console.error("Failed to load FAQs:",a),i.innerHTML=`
                <div class="text-center py-12">
                    <svg class="w-16 h-16 text-red-300 dark:text-red-600 mx-auto mb-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 8v4m0 4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z"/>
                    </svg>
                    <p class="text-red-500 dark:text-red-400">Failed to load FAQs. Please try again later.</p>
                </div>
            `}}}function H(i){let a={};return i.forEach(c=>{let u=c.category||"General";a[u]||(a[u]=[]),a[u].push(c)}),a}function U(i,a=""){let c=document.getElementById("faq-container");if(!c)return;let u=Object.keys(i);if(u.length===0){c.innerHTML=`
                <div class="text-center py-12">
                    <svg class="w-16 h-16 text-stone-300 dark:text-stone-600 mx-auto mb-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9.172 16.172a4 4 0 015.656 0M9 10h.01M15 10h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z"/>
                    </svg>
                    <p class="text-stone-500 dark:text-stone-400">No FAQs found${a?` for "${Templates.escapeHtml(a)}"`:""}.</p>
                </div>
            `;return}c.innerHTML=`
            <!-- Category Tabs -->
            <div class="mb-8 overflow-x-auto scrollbar-hide">
                <div class="flex gap-2 pb-2">
                    <button class="faq-category-btn px-4 py-2 bg-primary-600 dark:bg-amber-600 text-white rounded-full text-sm font-medium whitespace-nowrap transition-colors" data-category="all">
                        All
                    </button>
                    ${u.map(m=>`
                        <button class="faq-category-btn px-4 py-2 bg-stone-100 dark:bg-stone-800 hover:bg-stone-200 dark:hover:bg-stone-700 text-stone-600 dark:text-stone-300 rounded-full text-sm font-medium whitespace-nowrap transition-colors" data-category="${Templates.escapeHtml(m)}">
                            ${Templates.escapeHtml(m)}
                        </button>
                    `).join("")}
                </div>
            </div>

            <!-- FAQ Accordion -->
            <div id="faq-list" class="space-y-6">
                ${u.map(m=>`
                    <div class="faq-category" data-category="${Templates.escapeHtml(m)}">
                        <h2 class="text-lg font-semibold text-stone-900 dark:text-white mb-4">${Templates.escapeHtml(m)}</h2>
                        <div class="space-y-3">
                            ${i[m].map(b=>B(b,a)).join("")}
                        </div>
                    </div>
                `).join("")}
            </div>
        `,C(),k(),w(),A()}function B(i,a=""){let c=Templates.escapeHtml(i.question),u=i.answer;if(a){let m=new RegExp(`(${a.replace(/[.*+?^${}()|[\]\\]/g,"\\$&")})`,"gi");c=c.replace(m,'<mark class="bg-yellow-200 dark:bg-yellow-800">$1</mark>'),u=u.replace(m,'<mark class="bg-yellow-200 dark:bg-yellow-800">$1</mark>')}return`
            <div class="faq-item border border-stone-200 dark:border-stone-700 rounded-xl overflow-hidden bg-white dark:bg-stone-800" data-id="${i.id||""}">
                <button class="faq-trigger w-full px-6 py-4 text-left flex items-center justify-between hover:bg-stone-50 dark:hover:bg-stone-700/50 transition-colors">
                    <span class="font-medium text-stone-900 dark:text-white pr-4">${c}</span>
                    <svg class="faq-icon w-5 h-5 text-stone-500 dark:text-stone-400 flex-shrink-0 transition-transform" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M19 9l-7 7-7-7"/>
                    </svg>
                </button>
                <div class="faq-content hidden px-6 pb-4">
                    <div class="prose prose-sm dark:prose-invert max-w-none text-stone-600 dark:text-stone-300">
                        ${u}
                    </div>
                </div>
            </div>
        `}function C(){let i=document.querySelectorAll(".faq-category-btn"),a=document.querySelectorAll(".faq-category");i.forEach(c=>{c.addEventListener("click",()=>{i.forEach(m=>{m.classList.remove("bg-primary-600","dark:bg-amber-600","text-white"),m.classList.add("bg-stone-100","dark:bg-stone-800","text-stone-600","dark:text-stone-300")}),c.classList.add("bg-primary-600","dark:bg-amber-600","text-white"),c.classList.remove("bg-stone-100","dark:bg-stone-800","text-stone-600","dark:text-stone-300");let u=c.dataset.category;a.forEach(m=>{u==="all"||m.dataset.category===u?m.classList.remove("hidden"):m.classList.add("hidden")})})})}function k(){document.querySelectorAll(".faq-trigger").forEach(a=>{a.addEventListener("click",()=>{let c=a.closest(".faq-item"),u=c.querySelector(".faq-content"),m=c.querySelector(".faq-icon"),b=!u.classList.contains("hidden");document.querySelectorAll(".faq-item").forEach(v=>{v!==c&&(v.querySelector(".faq-content")?.classList.add("hidden"),v.querySelector(".faq-icon")?.classList.remove("rotate-180"))}),u.classList.toggle("hidden"),m.classList.toggle("rotate-180")})})}function S(){let i=document.getElementById("faq-search");if(!i)return;let a=null;i.addEventListener("input",u=>{let m=u.target.value.trim().toLowerCase();clearTimeout(a),a=setTimeout(()=>{if(document.querySelector(".accordion-toggle"))T(m);else if(n&&Object.keys(n).length>0){if(m.length<2){U(n);return}let v={};Object.entries(n).forEach(([P,D])=>{let J=D.filter(le=>le.question.toLowerCase().includes(m)||le.answer.toLowerCase().includes(m));J.length>0&&(v[P]=J)}),U(v,m)}},300)});let c=document.createElement("span");c.className="absolute right-3 top-1/2 -translate-y-1/2 text-xs text-stone-400 dark:text-stone-500 hidden md:block",c.textContent="Press / to search",i.parentElement&&(i.parentElement.style.position="relative",i.parentElement.appendChild(c),i.addEventListener("focus",()=>c.classList.add("hidden")),i.addEventListener("blur",()=>c.classList.remove("hidden")))}function T(i){let a=document.querySelectorAll(".faq-item"),c=document.querySelectorAll(".faq-category"),u=document.getElementById("no-results"),m=0;a.forEach(b=>{let v=b.querySelector(".accordion-toggle span, button span"),P=b.querySelector(".accordion-content"),D=v?v.textContent.toLowerCase():"",J=P?P.textContent.toLowerCase():"";!i||D.includes(i)||J.includes(i)?(b.classList.remove("hidden"),m++):b.classList.add("hidden")}),c.forEach(b=>{let v=b.querySelectorAll(".faq-item:not(.hidden)");b.classList.toggle("hidden",v.length===0)}),u&&u.classList.toggle("hidden",m>0)}function y(){n=[],e=[]}return{init:s,destroy:y}})();window.FAQPage=ar;Or=ar});var ir={};te(ir,{CategoryCard:()=>Vr});function Vr(n){let e=document.createElement("a");e.href=`/categories/${n.slug}/`,e.className="group block";let s=document.createElement("div");s.className="relative aspect-square rounded-xl overflow-hidden bg-gray-100";let o="";if(typeof n.image_url=="string"&&n.image_url?o=n.image_url:typeof n.image=="string"&&n.image?o=n.image:n.image&&typeof n.image=="object"?o=n.image.url||n.image.src||n.image_url||"":typeof n.banner_image=="string"&&n.banner_image?o=n.banner_image:n.banner_image&&typeof n.banner_image=="object"?o=n.banner_image.url||n.banner_image.src||"":typeof n.hero_image=="string"&&n.hero_image?o=n.hero_image:n.hero_image&&typeof n.hero_image=="object"?o=n.hero_image.url||n.hero_image.src||"":typeof n.thumbnail=="string"&&n.thumbnail?o=n.thumbnail:n.thumbnail&&typeof n.thumbnail=="object"&&(o=n.thumbnail.url||n.thumbnail.src||""),o){let _=document.createElement("img");_.src=o,_.alt=n.name||"",_.className="w-full h-full object-cover group-hover:scale-105 transition-transform duration-300",_.loading="lazy",_.decoding="async",_.onerror=L=>{try{_.remove();let A=document.createElement("div");A.className="w-full h-full flex items-center justify-center bg-gradient-to-br from-primary-100 to-primary-200",A.innerHTML='<svg class="w-12 h-12 text-primary-500" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M4 6a2 2 0 012-2h2a2 2 0 012 2v2a2 2 0 01-2 2H6a2 2 0 01-2-2V6zM14 6a2 2 0 012-2h2a2 2 0 012 2v2a2 2 0 01-2 2h-2a2 2 0 01-2-2V6zM4 16a2 2 0 012-2h2a2 2 0 012 2v2a2 2 0 01-2 2H6a2 2 0 01-2-2v-2zM14 16a2 2 0 012-2h2a2 2 0 012 2v2a2 2 0 01-2 2h-2a2 2 0 01-2-2v-2z"/></svg>',s.appendChild(A)}catch{}},s.appendChild(_)}else{let _=document.createElement("div");_.className="w-full h-full flex items-center justify-center bg-gradient-to-br from-primary-100 to-primary-200",_.innerHTML='<svg class="w-12 h-12 text-primary-500" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M4 6a2 2 0 012-2h2a2 2 0 012 2v2a2 2 0 01-2 2H6a2 2 0 01-2-2V6zM14 6a2 2 0 012-2h2a2 2 0 012 2v2a2 2 0 01-2 2h-2a2 2 0 01-2-2V6zM4 16a2 2 0 012-2h2a2 2 0 012 2v2a2 2 0 01-2 2H6a2 2 0 01-2-2v-2zM14 16a2 2 0 012-2h2a2 2 0 012 2v2a2 2 0 01-2 2h-2a2 2 0 01-2-2v-2z"/></svg>',s.appendChild(_)}let h=document.createElement("div");h.className="absolute inset-0 bg-gradient-to-t from-black/30 dark:from-black/60 to-transparent",s.appendChild(h),e.appendChild(s);let w=document.createElement("h3");if(w.className="mt-3 text-sm font-medium text-stone-900 group-hover:text-primary-600 transition-colors text-center dark:text-white",w.textContent=n.name,e.appendChild(w),n.product_count){let _=document.createElement("p");_.className="text-xs text-stone-600 dark:text-white/60 text-center",_.textContent=`${n.product_count} products`,e.appendChild(_)}return e}var lr=ee(()=>{});var cr={};te(cr,{default:()=>Dr});var dr,Dr,ur=ee(()=>{dr=(function(){"use strict";let n=null,e=0,s=null,o=null;async function h(){window.scrollTo(0,0),await Promise.all([H(),B(),k()]),y(),Y(),Z(),Promise.all([C(),E(),z()]).catch(a=>console.error("Failed to load secondary sections:",a)),setTimeout(()=>{w(),_(),L(),A()},2e3);try{S(),T()}catch(a){console.warn("Failed to load promotions/CTA:",a)}}function w(){let a=document.getElementById("live-visitors");if(!a)return;async function c(){try{let u=await window.ApiClient.get("/analytics/active-visitors/",{}),m=u.data||u;if(e=m.active_visitors||m.count||0,e===0){a.innerHTML="";return}a.innerHTML=`
                    <div class="flex items-center gap-2 px-3 py-1.5 bg-emerald-100 dark:bg-emerald-900/30 rounded-full">
                        <span class="relative flex h-2 w-2">
                            <span class="animate-ping absolute inline-flex h-full w-full rounded-full bg-emerald-400 opacity-75"></span>
                            <span class="relative inline-flex rounded-full h-2 w-2 bg-emerald-500"></span>
                        </span>
                        <span class="text-xs font-medium text-emerald-700 dark:text-emerald-300">${e} browsing now</span>
                    </div>
                `}catch(u){console.warn("Failed to fetch active visitors:",u),a.innerHTML=""}}c(),setInterval(c,8e3)}function _(){let a=[],c=0,u=0,m=10;async function b(){try{let P=await window.ApiClient.get("/analytics/recent-purchases/",{});if(a=P.data||P.purchases||[],a.length===0)return;setTimeout(()=>{v(),s=setInterval(()=>{u<m?v():clearInterval(s)},3e4)},1e4)}catch(P){console.warn("Failed to fetch recent purchases:",P)}}function v(){if(a.length===0||u>=m)return;let P=a[c];if(!P)return;let D=document.createElement("div");D.className="social-proof-popup fixed bottom-4 left-4 z-50 max-w-xs bg-white dark:bg-stone-800 rounded-xl shadow-2xl border border-stone-200 dark:border-stone-700 p-4 transform translate-y-full opacity-0 transition-all duration-500";let J=`
                <div class="flex items-start gap-3">
                    <div class="w-10 h-10 bg-emerald-100 dark:bg-emerald-900/30 rounded-full flex items-center justify-center flex-shrink-0">
                        <svg class="w-5 h-5 text-emerald-600 dark:text-emerald-400" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M5 13l4 4L19 7"/></svg>
                    </div>
                    <div>
                        <p class="text-sm font-medium text-stone-900 dark:text-white">${P.message}</p>
                        <p class="text-xs text-stone-400 dark:text-stone-500 mt-1">${P.time_ago}</p>
                    </div>
                </div>
            `;D.innerHTML=`
                ${J}
                <button class="absolute top-2 right-2 text-stone-400 hover:text-stone-600 dark:hover:text-stone-300" onclick="this.parentElement.remove()">
                    <svg class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M6 18L18 6M6 6l12 12"/></svg>
                </button>
            `,document.body.appendChild(D),u++,requestAnimationFrame(()=>{D.classList.remove("translate-y-full","opacity-0")}),setTimeout(()=>{D.classList.add("translate-y-full","opacity-0"),setTimeout(()=>D.remove(),500)},5e3),c=(c+1)%a.length,u>=m&&s&&clearInterval(s)}b()}function L(){let a=document.getElementById("recently-viewed-section"),c=document.getElementById("recently-viewed"),u=document.getElementById("clear-recently-viewed");if(!a||!c)return;let m=JSON.parse(localStorage.getItem("recentlyViewed")||"[]");if(m.length===0){a.classList.add("hidden");return}a.classList.remove("hidden"),c.innerHTML=m.slice(0,5).map(b=>{let v=null;return b.discount_percent&&b.discount_percent>0&&(v=`-${b.discount_percent}%`),ProductCard.render(b,{showBadge:!!v,badge:v,priceSize:"small"})}).join(""),ProductCard.bindEvents(c),u?.addEventListener("click",()=>{localStorage.removeItem("recentlyViewed"),a.classList.add("hidden"),Toast.success("Recently viewed items cleared")})}function A(){let a=document.getElementById("flash-sale-section"),c=document.getElementById("flash-sale-countdown");if(!a||!c)return;if(!localStorage.getItem("flashSaleEnd")){let v=new Date().getTime()+144e5;localStorage.setItem("flashSaleEnd",v.toString())}let m=parseInt(localStorage.getItem("flashSaleEnd"));function b(){let v=new Date().getTime(),P=m-v;if(P<=0){a.classList.add("hidden"),clearInterval(o),localStorage.removeItem("flashSaleEnd");return}a.classList.remove("hidden");let D=Math.floor(P%(1e3*60*60*24)/(1e3*60*60)),J=Math.floor(P%(1e3*60*60)/(1e3*60)),le=Math.floor(P%(1e3*60)/1e3);c.innerHTML=`
                <div class="flex items-center gap-2 text-white">
                    <span class="text-sm font-medium">Ends in:</span>
                    <div class="flex items-center gap-1">
                        <span class="bg-white/20 px-2 py-1 rounded font-mono font-bold">${D.toString().padStart(2,"0")}</span>
                        <span class="font-bold">:</span>
                        <span class="bg-white/20 px-2 py-1 rounded font-mono font-bold">${J.toString().padStart(2,"0")}</span>
                        <span class="font-bold">:</span>
                        <span class="bg-white/20 px-2 py-1 rounded font-mono font-bold">${le.toString().padStart(2,"0")}</span>
                    </div>
                </div>
            `}b(),o=setInterval(b,1e3)}function Y(){let a=document.querySelectorAll("[data-animate]");if(!a.length)return;let c=new IntersectionObserver(u=>{u.forEach(m=>{if(m.isIntersecting){let b=m.target.dataset.animate||"fadeInUp";m.target.classList.add("animate-"+b),m.target.classList.remove("opacity-0"),c.unobserve(m.target)}})},{threshold:.1,rootMargin:"0px 0px -50px 0px"});a.forEach(u=>{u.classList.add("opacity-0"),c.observe(u)})}function Z(){document.addEventListener("click",async a=>{let c=a.target.closest("[data-quick-view]");if(!c)return;let u=c.dataset.quickView;if(!u)return;a.preventDefault();let m=document.createElement("div");m.id="quick-view-modal",m.className="fixed inset-0 z-50 flex items-center justify-center p-4",m.innerHTML=`
                <div class="absolute inset-0 bg-black/50 backdrop-blur-sm" onclick="document.getElementById('quick-view-modal').remove()"></div>
                <div class="relative bg-white dark:bg-stone-800 rounded-2xl shadow-2xl max-w-4xl w-full max-h-[90vh] overflow-auto">
                    <button class="absolute top-4 right-4 z-10 w-10 h-10 bg-stone-100 dark:bg-stone-700 rounded-full flex items-center justify-center hover:bg-stone-200 dark:hover:bg-stone-600 transition-colors" onclick="document.getElementById('quick-view-modal').remove()">
                        <svg class="w-5 h-5 text-stone-600 dark:text-stone-300" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M6 18L18 6M6 6l12 12"/></svg>
                    </button>
                    <div class="p-8 flex items-center justify-center min-h-[400px]">
                        <div class="animate-spin rounded-full h-12 w-12 border-b-2 border-primary-600"></div>
                    </div>
                </div>
            `,document.body.appendChild(m);try{let b=await ProductsApi.getProduct(u),v=b.data||b,P=m.querySelector(".relative");P.innerHTML=`
                    <button class="absolute top-4 right-4 z-10 w-10 h-10 bg-stone-100 dark:bg-stone-700 rounded-full flex items-center justify-center hover:bg-stone-200 dark:hover:bg-stone-600 transition-colors" onclick="document.getElementById('quick-view-modal').remove()">
                        <svg class="w-5 h-5 text-stone-600 dark:text-stone-300" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M6 18L18 6M6 6l12 12"/></svg>
                    </button>
                    <div class="grid md:grid-cols-2 gap-8 p-8">
                        <div class="aspect-square rounded-xl overflow-hidden bg-stone-100 dark:bg-stone-700">
                            <img src="${v.primary_image||v.image||"/static/images/placeholder.jpg"}" alt="${Templates.escapeHtml(v.name)}" class="w-full h-full object-cover">
                        </div>
                        <div class="flex flex-col">
                            <h2 class="text-2xl font-bold text-stone-900 dark:text-white mb-2">${Templates.escapeHtml(v.name)}</h2>
                            <div class="flex items-center gap-2 mb-4">
                                <div class="flex text-amber-400">
                                    ${'<svg class="w-4 h-4" fill="currentColor" viewBox="0 0 20 20"><path d="M9.049 2.927c.3-.921 1.603-.921 1.902 0l1.286 3.967a1 1 0 00.95.69h4.178c.969 0 1.371 1.24.588 1.81l-3.385 2.46a1 1 0 00-.364 1.118l1.287 3.966c.3.922-.755 1.688-1.54 1.118l-3.385-2.46a1 1 0 00-1.175 0l-3.385 2.46c-.784.57-1.838-.196-1.54-1.118l1.287-3.966a1 1 0 00-.364-1.118l-3.385-2.46c-.783-.57-.38-1.81.588-1.81h4.178a1 1 0 00.95-.69l1.286-3.967z"/></svg>'.repeat(Math.round(v.rating||4))}
                                </div>
                                <span class="text-sm text-stone-500 dark:text-stone-400">(${v.review_count||0} reviews)</span>
                            </div>
                            <div class="mb-6">
                                ${v.sale_price||v.discounted_price?`
                                    <span class="text-3xl font-bold text-primary-600 dark:text-amber-400">${Templates.formatPrice(v.sale_price||v.discounted_price)}</span>
                                    <span class="text-lg text-stone-400 line-through ml-2">${Templates.formatPrice(v.price)}</span>
                                `:`
                                    <span class="text-3xl font-bold text-stone-900 dark:text-white">${Templates.formatPrice(v.price)}</span>
                                `}
                            </div>
                            <p class="text-stone-600 dark:text-stone-400 mb-6 line-clamp-3">${Templates.escapeHtml(v.short_description||v.description||"")}</p>
                            <div class="mt-auto space-y-3">
                                <button class="w-full py-3 px-6 bg-primary-600 hover:bg-primary-700 text-white font-semibold rounded-xl transition-colors" onclick="CartApi.addItem(${v.id}, 1).then(() => { Toast.success('Added to cart'); document.getElementById('quick-view-modal').remove(); })">
                                    Add to Cart
                                </button>
                                <a href="/products/${v.slug||v.id}/" class="block w-full py-3 px-6 border-2 border-stone-200 dark:border-stone-600 text-stone-700 dark:text-stone-300 font-semibold rounded-xl text-center hover:bg-stone-50 dark:hover:bg-stone-700 transition-colors">
                                    View Full Details
                                </a>
                            </div>
                        </div>
                    </div>
                `}catch(b){console.error("Failed to load product:",b),m.remove(),Toast.error("Failed to load product details")}})}async function z(){let a=document.getElementById("testimonials-grid");if(a){Loader.show(a,"skeleton");try{let c=await ProductsApi.getReviews(null,{pageSize:6,orderBy:"-rating"}),u=c?.data?.results||c?.data||c?.results||[];if(a.innerHTML="",!u.length){a.innerHTML='<p class="text-gray-500 text-center py-8">No user reviews available.</p>';return}u=u.slice(0,6),u.forEach(m=>{let b=document.createElement("div");b.className="rounded-2xl bg-white dark:bg-stone-800 shadow p-6 flex flex-col gap-3",b.innerHTML=`
                        <div class="flex items-center gap-3 mb-2">
                            <div class="w-10 h-10 rounded-full bg-primary-100 dark:bg-stone-700 flex items-center justify-center text-lg font-bold text-primary-700 dark:text-amber-400">
                                ${m.user?.first_name?.[0]||m.user?.username?.[0]||"?"}
                            </div>
                            <div>
                                <div class="font-semibold text-gray-900 dark:text-white">${m.user?.first_name||m.user?.username||"Anonymous"}</div>
                                <div class="text-xs text-gray-500 dark:text-stone-400">${m.created_at?new Date(m.created_at).toLocaleDateString():""}</div>
                            </div>
                        </div>
                        <div class="flex gap-1 mb-1">
                            ${'<svg class="w-4 h-4 text-amber-400" fill="currentColor" viewBox="0 0 20 20"><path d="M9.049 2.927c.3-.921 1.603-.921 1.902 0l1.286 3.967a1 1 0 00.95.69h4.178c.969 0 1.371 1.24.588 1.81l-3.385 2.46a1 1 0 00-.364 1.118l1.287 3.966c.3.922-.755 1.688-1.54 1.118l-3.385-2.46a1 1 0 00-1.175 0l-3.385 2.46c-.784.57-1.838-.196-1.54-1.118l1.287-3.966a1 1 0 00-.364-1.118l-3.385-2.46c-.783-.57-.38-1.81.588-1.81h4.178a1 1 0 00.95-.69l1.286-3.967z"/></svg>'.repeat(Math.round(m.rating||5))}
                        </div>
                        <div class="text-gray-800 dark:text-stone-200 text-base mb-2">${Templates.escapeHtml(m.title||"")}</div>
                        <div class="text-gray-600 dark:text-stone-400 text-sm">${Templates.escapeHtml(m.content||"")}</div>
                    `,a.appendChild(b)})}catch(c){console.error("Failed to load testimonials:",c),a.innerHTML='<p class="text-red-500 text-center py-8">Failed to load reviews. Please try again later.</p>'}}}async function E(){let a=document.getElementById("best-sellers");if(!a)return;let c=a.querySelector(".products-grid")||a;Loader.show(c,"skeleton");try{let u=await ProductsApi.getProducts({bestseller:!0,pageSize:5}),m=u.data?.results||u.data||u.results||[];if(m.length===0){c.innerHTML='<p class="text-gray-500 text-center py-8">No best sellers available.</p>';return}c.innerHTML=m.map(b=>{let v=null;return b.discount_percent&&b.discount_percent>0&&(v=`-${b.discount_percent}%`),ProductCard.render(b,{showBadge:!!v,badge:v,priceSize:"small"})}).join(""),ProductCard.bindEvents(c)}catch(u){console.error("Failed to load best sellers:",u),c.innerHTML='<p class="text-red-500 text-center py-8">Failed to load products. Please try again later.</p>'}}async function H(){let a=document.getElementById("hero-slider");if(a)try{let c=await PagesApi.getBanners("home_hero"),u=c.data?.results||c.data||c.results||[];if(u.length===0){a.innerHTML="";return}a.innerHTML=`
                <div class="relative overflow-hidden w-full h-[55vh] sm:h-[70vh] md:h-[80vh]">
                    <div class="hero-slides relative w-full h-full">
                        ${u.map((m,b)=>`
                            <div class="hero-slide ${b===0?"":"hidden"} w-full h-full" data-index="${b}">
                                <a href="${m.link_url||"#"}" class="block relative w-full h-full">
                                    <img 
                                        src="${m.image}" 
                                        alt="${Templates.escapeHtml(m.title||"")}"
                                        class="absolute inset-0 w-full h-full object-cover"
                                        loading="${b===0?"eager":"lazy"}"
                                        decoding="async"
                                    >
                                    ${m.title||m.subtitle?`
                                        <div class="absolute inset-0 bg-gradient-to-r from-black/60 via-black/30 to-transparent flex items-center">
                                            <div class="px-8 md:px-16 max-w-xl">
                                                ${m.title?`<h2 class="text-2xl sm:text-3xl md:text-5xl font-bold text-white mb-4">${Templates.escapeHtml(m.title)}</h2>`:""}
                                                ${m.subtitle?`<p class="text-sm sm:text-lg text-white/90 mb-6">${Templates.escapeHtml(m.subtitle)}</p>`:""}
                                                ${m.link_text||m.button_text?`
                                                    <span class="inline-flex items-center px-6 py-3 bg-white text-gray-900 font-semibold rounded-lg hover:bg-gray-100 transition-colors">
                                                        ${Templates.escapeHtml(m.link_text||m.button_text)}
                                                        <svg class="w-5 h-5 ml-2" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                                            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M17 8l4 4m0 0l-4 4m4-4H3"/>
                                                        </svg>
                                                    </span>
                                                `:""}
                                            </div>
                                        </div>
                                    `:""}
                                </a>
                            </div>
                        `).join("")}
                    </div>
                    ${u.length>1?`
                        <button class="hero-prev absolute left-4 top-1/2 -translate-y-1/2 w-10 h-10 bg-white/30 dark:bg-stone-800/30 hover:bg-white/40 dark:hover:bg-stone-700/40 rounded-full text-stone-900 dark:text-stone-100 flex items-center justify-center shadow-lg transition-colors focus:outline-none focus-visible:ring-2 focus-visible:ring-amber-500" aria-label="Previous slide">
                            <svg class="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24" aria-hidden="true">
                                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M15 19l-7-7 7-7"/>
                            </svg>
                        </button>
                        <button class="hero-next absolute right-4 top-1/2 -translate-y-1/2 w-10 h-10 bg-white/30 dark:bg-stone-800/30 hover:bg-white/40 dark:hover:bg-stone-700/40 rounded-full text-stone-900 dark:text-stone-100 flex items-center justify-center shadow-lg transition-colors focus:outline-none focus-visible:ring-2 focus-visible:ring-amber-500" aria-label="Next slide">
                            <svg class="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24" aria-hidden="true">
                                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9 5l7 7-7 7"/>
                            </svg>
                        </button>
                        <div class="hero-dots absolute bottom-4 left-1/2 -translate-x-1/2 flex gap-2">
                            ${u.map((m,b)=>`
                                <button class="w-3 h-3 rounded-full transition-colors ${b===0?"bg-white dark:bg-stone-200":"bg-white/50 dark:bg-stone-600/60 hover:bg-white/75 dark:hover:bg-stone-500/80"}" data-slide="${b}" aria-label="Go to slide ${b+1}"></button>
                            `).join("")}
                        </div>
                    `:""}
                </div>
            `,u.length>1&&U(u.length)}catch(c){console.warn("Hero banners unavailable:",c?.status||c)}}function U(a){let c=0,u=document.querySelectorAll(".hero-slide"),m=document.querySelectorAll(".hero-dots button"),b=document.querySelector(".hero-prev"),v=document.querySelector(".hero-next");function P(J){u[c].classList.add("hidden"),m[c]?.classList.remove("bg-stone-100"),m[c]?.classList.add("bg-white/50"),c=(J+a)%a,u[c].classList.remove("hidden"),m[c]?.classList.add("bg-stone-100"),m[c]?.classList.remove("bg-white/50")}b?.addEventListener("click",()=>{P(c-1),D()}),v?.addEventListener("click",()=>{P(c+1),D()}),m.forEach((J,le)=>{J.addEventListener("click",()=>{P(le),D()})});function D(){n&&clearInterval(n),n=setInterval(()=>P(c+1),5e3)}try{let J=document.querySelector(".hero-slides"),le=0;J?.addEventListener("touchstart",be=>{le=be.touches[0].clientX},{passive:!0}),J?.addEventListener("touchend",be=>{let fe=be.changedTouches[0].clientX-le;Math.abs(fe)>50&&(fe<0?P(c+1):P(c-1),D())})}catch{}D()}async function B(){let a=document.getElementById("featured-products");if(!a)return;let c=a.querySelector(".products-grid")||a;Loader.show(c,"skeleton");try{let u=await ProductsApi.getFeatured(8),m=u.data?.results||u.data||u.results||[];if(m.length===0){c.innerHTML='<p class="text-gray-500 text-center py-8">No featured products available.</p>';return}c.innerHTML=m.map(b=>{let v=null;return b.discount_percent&&b.discount_percent>0&&(v=`-${b.discount_percent}%`),ProductCard.render(b,{showBadge:!!v,badge:v,priceSize:"small"})}).join(""),ProductCard.bindEvents(c)}catch(u){console.error("Failed to load featured products:",u),c.innerHTML='<p class="text-red-500 text-center py-8">Failed to load products. Please try again later.</p>'}}async function C(){let a=document.getElementById("categories-showcase");if(a){Loader.show(a,"skeleton");try{try{window.ApiClient?.clearCache("/api/v1/catalog/categories/")}catch{}let c=await window.ApiClient.get("/catalog/categories/",{page_size:6,is_featured:!0},{useCache:!1}),u=c.data?.results||c.data||c.results||[];if(u.length===0){a.innerHTML="";return}let m;try{m=(await Promise.resolve().then(()=>(lr(),ir))).CategoryCard}catch(b){console.error("Failed to import CategoryCard:",b);return}a.innerHTML="",u.forEach(b=>{let v=m(b);try{let P=v.querySelector("img");console.info("[Home] card image for",b.name,P?P.src:"NO_IMAGE")}catch{}a.appendChild(v)}),a.classList.add("grid","grid-cols-2","sm:grid-cols-2","md:grid-cols-3","lg:grid-cols-6","gap-4","lg:gap-6")}catch(c){console.error("Failed to load categories:",c),a.innerHTML=""}}}async function k(){let a=document.getElementById("new-arrivals");if(!a)return;let c=a.querySelector(".products-grid")||a;Loader.show(c,"skeleton");try{let u=await ProductsApi.getNewArrivals(4),m=u.data?.results||u.data||u.results||[];if(m.length===0){c.innerHTML='<p class="text-gray-500 text-center py-8">No new products available.</p>';return}c.innerHTML=m.map(b=>{let v=null;return b.discount_percent&&b.discount_percent>0&&(v=`-${b.discount_percent}%`),ProductCard.render(b,{showBadge:!!v,badge:v,priceSize:"small"})}).join(""),ProductCard.bindEvents(c)}catch(u){console.error("Failed to load new arrivals:",u),c.innerHTML='<p class="text-red-500 text-center py-8">Failed to load products.</p>'}}async function S(){let a=document.getElementById("promotions-banner")||document.getElementById("promotion-banners");if(a)try{let c=await PagesApi.getPromotions(),u=c?.data?.results??c?.results??c?.data??[];if(Array.isArray(u)||(u&&typeof u=="object"?u=Array.isArray(u.items)?u.items:[u]:u=[]),u.length===0){a.innerHTML="";return}let m=u[0]||{};a.innerHTML=`
                <div class="bg-gradient-to-r from-primary-600 to-primary-700 rounded-2xl overflow-hidden">
                    <div class="px-6 py-8 md:px-12 md:py-12 flex flex-col md:flex-row items-center justify-between gap-6">
                        <div class="text-center md:text-left">
                            <span class="inline-block px-3 py-1 bg-white/20 text-white text-sm font-medium rounded-full mb-3">
                                Limited Time Offer
                            </span>
                            <h3 class="text-2xl md:text-3xl font-bold text-white mb-2">
                                ${Templates.escapeHtml(m.title||m.name||"")}
                            </h3>
                            ${m.description?`
                                <p class="text-white/90 max-w-lg">${Templates.escapeHtml(m.description)}</p>
                            `:""}
                            ${m.discount_value?`
                                <p class="text-3xl font-bold text-white mt-4">
                                    ${m.discount_type==="percentage"?`${m.discount_value}% OFF`:`Save ${Templates.formatPrice(m.discount_value)}`}
                                </p>
                            `:""}
                        </div>
                        <div class="flex flex-col items-center gap-4">
                            ${m.code?`
                                <div class="bg-white/10 backdrop-blur-sm px-6 py-3 rounded-lg border-2 border-dashed border-white/30">
                                    <p class="text-sm text-white/80 mb-1">Use code:</p>
                                    <p class="text-2xl font-mono font-bold text-white tracking-wider">${Templates.escapeHtml(m.code)}</p>
                                </div>
                            `:""}
                            <a href="/products/?promotion=${m.id||""}" class="inline-flex items-center px-6 py-3 bg-stone-100 text-primary-600 font-semibold rounded-lg hover:bg-gray-100 transition-colors">
                                Shop Now
                                <svg class="w-5 h-5 ml-2" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                    <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M17 8l4 4m0 0l-4 4m4-4H3"/>
                                </svg>
                            </a>
                        </div>
                    </div>
                </div>
            `}catch(c){console.warn("Promotions unavailable:",c?.status||c),a.innerHTML=""}}async function T(){let a=document.getElementById("custom-order-cta");if(!a||a.dataset?.loaded)return;a.dataset.loaded="1",a.innerHTML=`
            <div class="container mx-auto px-4">
                <div class="max-w-full w-full mx-auto rounded-3xl p-6 md:p-10">
                    <div class="animate-pulse">
                        <div class="h-6 w-1/3 bg-gray-200 dark:bg-stone-700 rounded mb-4"></div>
                        <div class="h-10 w-full bg-gray-200 dark:bg-stone-700 rounded mb-4"></div>
                        <div class="h-44 bg-gray-200 dark:bg-stone-800 rounded"></div>
                    </div>
                </div>
            </div>
        `;let c=window.BUNORAA_ROUTES||{},u=c.preordersWizard||"/preorders/create/",m=c.preordersLanding||"/preorders/";try{let b=[];if(typeof PreordersApi<"u"&&PreordersApi.getCategories)try{let v=await PreordersApi.getCategories({featured:!0,pageSize:4});b=v?.data?.results||v?.data||v?.results||[]}catch(v){console.warn("Pre-order categories unavailable:",v)}a.innerHTML=`
                <div class="container mx-auto px-4 relative">
                    <div class="max-w-full w-full mx-auto rounded-3xl shadow-lg overflow-hidden bg-white dark:bg-neutral-900 p-6 md:p-10 border border-stone-100 dark:border-stone-700">
                      <div class="grid lg:grid-cols-2 gap-12 items-center">
                        <div class="text-stone-900 dark:text-white">
                            <span class="inline-flex items-center gap-2 px-4 py-2 rounded-full bg-amber-100/40 dark:bg-amber-700/20 text-xs uppercase tracking-[0.2em] mb-6 text-amber-800 dark:text-white">
                                <span class="w-2 h-2 bg-purple-400 rounded-full animate-pulse"></span>
                                Made Just For You
                            </span>
                            <h2 class="text-3xl lg:text-5xl font-display font-bold mb-6 leading-tight text-stone-900 dark:text-white">Create Your Perfect Custom Order</h2>
                            <p class="text-stone-700 dark:text-white/80 text-lg mb-8 max-w-xl">Have a unique vision? Our skilled artisans will bring your ideas to life. From personalized gifts to custom designs, we craft exactly what you need.</p>
                            <div class="grid sm:grid-cols-3 gap-4 mb-8">
                                <div class="flex items-center gap-3 bg-white/5 dark:bg-stone-800/40 backdrop-blur-sm rounded-xl p-4 border border-stone-100 dark:border-stone-700">
                                    <div class="w-10 h-10 bg-purple-500/30 rounded-lg flex items-center justify-center">
                                        <svg class="w-5 h-5 text-purple-800" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M11 5H6a2 2 0 00-2 2v11a2 2 0 002 2h11a2 2 0 002-2v-5m-1.414-9.414a2 2 0 112.828 2.828L11.828 15H9v-2.828l8.586-8.586z"/></svg>
                                    </div>
                                    <div>
                                        <p class="text-sm font-semibold text-stone-900 dark:text-white">Custom Design</p>
                                        <p class="text-xs text-stone-600 dark:text-white/60">Your vision, our craft</p>
                                    </div>
                                </div>
                                <div class="flex items-center gap-3 bg-white/5 dark:bg-stone-800/40 backdrop-blur-sm rounded-xl p-4 border border-stone-100 dark:border-stone-700">
                                    <div class="w-10 h-10 bg-indigo-500/30 rounded-lg flex items-center justify-center">
                                        <svg class="w-5 h-5 text-indigo-700 dark:text-indigo-200" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M8 12h.01M12 12h.01M16 12h.01M21 12c0 4.418-4.03 8-9 8a9.863 9.863 0 01-4.255-.949L3 20l1.395-3.72C3.512 15.042 3 13.574 3 12c0-4.418 4.03-8 9-8s9 3.582 9 8z"/></svg>
                                    </div>
                                    <div>
                                        <p class="text-sm font-semibold text-stone-900 dark:text-white">Direct Chat</p>
                                        <p class="text-xs text-stone-600 dark:text-white/60">Talk to artisans</p>
                                    </div>
                                </div>
                                <div class="flex items-center gap-3 bg-white/5 dark:bg-stone-800/40 backdrop-blur-sm rounded-xl p-4 border border-stone-100 dark:border-stone-700">
                                    <div class="w-10 h-10 bg-pink-500/30 rounded-lg flex items-center justify-center">
                                        <svg class="w-5 h-5 text-pink-700 dark:text-pink-200" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z"/></svg>
                                    </div>
                                    <div>
                                        <p class="text-sm font-semibold text-stone-900 dark:text-white">Quality Assured</p>
                                        <p class="text-xs text-stone-600 dark:text-white/60">Satisfaction guaranteed</p>
                                    </div>
                                </div>
                            </div>
                            ${b.length>0?`
                                <div class="mb-8">
                                    <p class="text-stone-600 dark:text-white/60 text-sm mb-3">Popular categories:</p>
                                    <div class="flex flex-wrap gap-2">
                                        ${b.slice(0,4).map(v=>`
                                            <a href="${m}category/${v.slug}/" class="inline-flex items-center gap-2 px-4 py-2 bg-white/10 dark:bg-stone-800/30 hover:bg-white/20 dark:hover:bg-stone-700 rounded-full text-sm text-stone-900 dark:text-white transition-colors">
                                                ${v.icon?`<span>${v.icon}</span>`:""}
                                                ${Templates.escapeHtml(v.name)}
                                            </a>
                                        `).join("")}
                                    </div>
                                </div>
                            `:""}
                            <div class="flex flex-wrap gap-4">
                                <a href="${u}" class="cta-unlock inline-flex items-center gap-2 px-8 py-4 bg-amber-600 text-white font-bold rounded-xl shadow-lg hover:shadow-xl hover:text-black dark:hover:text-black transition-colors group dark:bg-amber-600 dark:text-white">
                                    Start Your Custom Order
                                    <svg class="w-5 h-5 group-hover:translate-x-1 transition-transform" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M17 8l4 4m0 0l-4 4m4-4H3"/></svg>
                                </a>
                                <a href="${m}" class="inline-flex items-center gap-2 px-8 py-4 bg-transparent text-stone-900 dark:text-white font-semibold rounded-xl border-2 border-stone-200 dark:border-stone-700 hover:bg-stone-100 dark:hover:bg-stone-800 transition-all">
                                    Learn More
                                </a>
                            </div>
                        </div>
                        <div class="hidden lg:block">
                            <div class="relative">
                                <div class="absolute -inset-4 bg-gradient-to-r from-purple-500/30 to-indigo-500/30 rounded-3xl blur-2xl"></div>
                                <div class="relative bg-white/5 dark:bg-stone-800/40 backdrop-blur-md rounded-3xl p-8 border border-stone-100 dark:border-stone-700">
                                    <div class="space-y-6">
                                        <div class="flex items-start gap-4">
                                            <div class="w-12 h-12 bg-purple-600 rounded-xl flex items-center justify-center flex-shrink-0 text-white text-xl font-bold shadow-sm ring-1 ring-stone-100 dark:ring-stone-700">1</div>
                                            <div>
                                                <h4 class="text-stone-900 dark:text-white font-semibold mb-1">Choose Category</h4>
                                                <p class="text-stone-600 dark:text-white/60 text-sm">Select from custom apparel, gifts, home decor & more</p>
                                            </div>
                                        </div>
                                        <div class="flex items-start gap-4">
                                            <div class="w-12 h-12 bg-indigo-600 rounded-xl flex items-center justify-center flex-shrink-0 text-white text-xl font-bold shadow-sm ring-1 ring-stone-100 dark:ring-stone-700">2</div>
                                            <div>
                                                <h4 class="text-stone-900 dark:text-white font-semibold mb-1">Share Your Vision</h4>
                                                <p class="text-stone-600 dark:text-white/60 text-sm">Upload designs, describe your requirements</p>
                                            </div>
                                        </div>
                                        <div class="flex items-start gap-4">
                                            <div class="w-12 h-12 bg-amber-600 rounded-xl flex items-center justify-center flex-shrink-0 text-white text-xl font-bold shadow-sm ring-1 ring-stone-100 dark:ring-stone-700">3</div>
                                            <div>
                                                <h4 class="text-stone-900 dark:text-white font-semibold mb-1">Get Your Quote</h4>
                                                <p class="text-stone-600 dark:text-white/60 text-sm">Receive pricing and timeline from our team</p>
                                            </div>
                                        </div>
                                        <div class="flex items-start gap-4">
                                            <div class="w-12 h-12 bg-emerald-600 rounded-xl flex items-center justify-center flex-shrink-0 text-white text-xl font-bold shadow-sm ring-1 ring-stone-100 dark:ring-stone-700">4</div>
                                            <div>
                                                <h4 class="text-stone-900 dark:text-white font-semibold mb-1">We Create & Deliver</h4>
                                                <p class="text-stone-600 dark:text-white/60 text-sm">Track progress and receive your masterpiece</p>
                                            </div>
                                        </div>
                                    </div>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
            `}catch(b){console.warn("Custom order CTA failed to load:",b),a.innerHTML=`
                <div class="container mx-auto px-4 text-center text-stone-900 dark:text-white">
                    <h2 class="text-3xl lg:text-4xl font-display font-bold mb-4 text-stone-900 dark:text-white">Create Your Perfect Custom Order</h2>
                    <p class="text-stone-700 dark:text-white/80 mb-8 max-w-2xl mx-auto">Have a unique vision? Our skilled artisans will bring your ideas to life.</p>
                    <a href="${u}" class="cta-unlock inline-flex items-center gap-2 px-8 py-4 bg-amber-600 text-white font-bold rounded-xl shadow-lg hover:shadow-xl hover:text-black dark:hover:text-black transition-colors group dark:bg-amber-600 dark:text-white">
                        Start Your Custom Order
                    </a>
                </div>
            `}}function y(){let a=document.getElementById("newsletter-form")||document.getElementById("newsletter-form-home");a&&a.addEventListener("submit",async c=>{c.preventDefault();let u=a.querySelector('input[type="email"]'),m=a.querySelector('button[type="submit"]'),b=u?.value?.trim();if(!b){Toast.error("Please enter your email address.");return}let v=m.textContent;m.disabled=!0,m.innerHTML='<svg class="animate-spin h-5 w-5 mx-auto" fill="none" viewBox="0 0 24 24"><circle class="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" stroke-width="4"></circle><path class="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path></svg>';try{await SupportApi.submitContactForm({email:b,type:"newsletter"}),Toast.success("Thank you for subscribing!"),u.value=""}catch(P){Toast.error(P.message||"Failed to subscribe. Please try again.")}finally{m.disabled=!1,m.textContent=v}})}function i(){n&&(clearInterval(n),n=null),s&&(clearInterval(s),s=null),o&&(clearInterval(o),o=null),document.getElementById("quick-view-modal")?.remove(),document.querySelectorAll(".social-proof-popup").forEach(a=>a.remove())}return{init:h,destroy:i,initRecentlyViewed:L,initFlashSaleCountdown:A}})();window.HomePage=dr;Dr=dr});var pr={};te(pr,{default:()=>Ur});var mr,Ur,gr=ee(()=>{mr=(function(){"use strict";let n=1,e="all";async function s(){if(!AuthGuard.protectPage())return;let E=o();E?await L(E):(await h(),Z())}function o(){let H=window.location.pathname.match(/\/orders\/([^\/]+)/);return H?H[1]:null}async function h(){let E=document.getElementById("orders-list");if(E){Loader.show(E,"skeleton");try{let H={page:n,limit:10};e!=="all"&&(H.status=e);let U=await OrdersApi.getAll(H),B=U.data||[],C=U.meta||{};w(B,C)}catch(H){console.error("Failed to load orders:",H),E.innerHTML='<p class="text-red-500 text-center py-8">Failed to load orders.</p>'}}}function w(E,H){let U=document.getElementById("orders-list");if(!U)return;if(E.length===0){U.innerHTML=`
                <div class="text-center py-16">
                    <svg class="w-24 h-24 text-gray-300 mx-auto mb-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path stroke-linecap="round" stroke-linejoin="round" stroke-width="1.5" d="M9 5H7a2 2 0 00-2 2v12a2 2 0 002 2h10a2 2 0 002-2V7a2 2 0 00-2-2h-2M9 5a2 2 0 002 2h2a2 2 0 002-2M9 5a2 2 0 012-2h2a2 2 0 012 2"/>
                    </svg>
                    <h2 class="text-2xl font-bold text-gray-900 mb-2">No orders yet</h2>
                    <p class="text-gray-600 mb-8">When you place an order, it will appear here.</p>
                    <a href="/products/" class="inline-flex items-center px-6 py-3 bg-primary-600 text-white font-semibold rounded-lg hover:bg-primary-700 transition-colors">
                        Start Shopping
                    </a>
                </div>
            `;return}U.innerHTML=`
            <div class="space-y-4">
                ${E.map(C=>_(C)).join("")}
            </div>
            ${H.total_pages>1?`
                <div id="orders-pagination" class="mt-8">${Pagination.render({currentPage:H.current_page||n,totalPages:H.total_pages,totalItems:H.total})}</div>
            `:""}
        `,document.getElementById("orders-pagination")?.addEventListener("click",C=>{let k=C.target.closest("[data-page]");k&&(n=parseInt(k.dataset.page),h(),window.scrollTo({top:0,behavior:"smooth"}))})}function _(E){let U={pending:"bg-yellow-100 text-yellow-700 dark:bg-yellow-900/30 dark:text-yellow-400",processing:"bg-blue-100 text-blue-700 dark:bg-blue-900/30 dark:text-blue-400",shipped:"bg-indigo-100 text-indigo-700 dark:bg-indigo-900/30 dark:text-indigo-400",delivered:"bg-green-100 text-green-700 dark:bg-green-900/30 dark:text-green-400",cancelled:"bg-red-100 text-red-700 dark:bg-red-900/30 dark:text-red-400",refunded:"bg-gray-100 text-gray-700 dark:bg-gray-800 dark:text-gray-400"}[E.status]||"bg-gray-100 text-gray-700 dark:bg-gray-800 dark:text-gray-400",B=E.items||[],C=B.slice(0,3),k=B.length-3,S=["pending","processing","shipped","delivered"],T=S.indexOf(E.status),y=E.status==="cancelled"||E.status==="refunded";return`
            <div class="bg-white dark:bg-stone-800 rounded-xl shadow-sm border border-gray-100 dark:border-stone-700 overflow-hidden">
                <div class="px-6 py-4 border-b border-gray-100 dark:border-stone-700 flex flex-wrap items-center justify-between gap-4">
                    <div>
                        <p class="text-sm text-gray-500 dark:text-stone-400">Order #${Templates.escapeHtml(E.order_number||E.id)}</p>
                        <p class="text-sm text-gray-500 dark:text-stone-400">Placed on ${Templates.formatDate(E.created_at)}</p>
                    </div>
                    <div class="flex items-center gap-4">
                        <span class="px-3 py-1 rounded-full text-sm font-medium ${U}">
                            ${Templates.escapeHtml(E.status_display||E.status)}
                        </span>
                        <a href="/orders/${E.id}/" class="text-primary-600 dark:text-amber-400 hover:text-primary-700 dark:hover:text-amber-300 font-medium text-sm">
                            View Details
                        </a>
                    </div>
                </div>
                
                <!-- Visual Progress Bar -->
                ${y?"":`
                    <div class="px-6 py-3 bg-stone-50 dark:bg-stone-900/50 border-b border-gray-100 dark:border-stone-700">
                        <div class="flex items-center justify-between relative">
                            <div class="absolute left-0 right-0 top-1/2 h-1 bg-stone-200 dark:bg-stone-700 -translate-y-1/2 rounded-full"></div>
                            <div class="absolute left-0 top-1/2 h-1 bg-primary-500 dark:bg-amber-500 -translate-y-1/2 rounded-full transition-all duration-500" style="width: ${Math.max(0,T/(S.length-1)*100)}%"></div>
                            ${S.map((i,a)=>`
                                <div class="relative z-10 flex flex-col items-center">
                                    <div class="w-6 h-6 rounded-full flex items-center justify-center text-xs font-medium ${a<=T?"bg-primary-500 dark:bg-amber-500 text-white":"bg-stone-200 dark:bg-stone-700 text-stone-500 dark:text-stone-400"}">
                                        ${a<T?'<svg class="w-3 h-3" fill="currentColor" viewBox="0 0 20 20"><path fill-rule="evenodd" d="M16.707 5.293a1 1 0 010 1.414l-8 8a1 1 0 01-1.414 0l-4-4a1 1 0 011.414-1.414L8 12.586l7.293-7.293a1 1 0 011.414 0z" clip-rule="evenodd"/></svg>':a+1}
                                    </div>
                                    <span class="text-xs text-stone-500 dark:text-stone-400 mt-1 capitalize hidden sm:block">${i}</span>
                                </div>
                            `).join("")}
                        </div>
                    </div>
                `}
                
                <div class="p-6">
                    <div class="flex flex-wrap gap-4">
                        ${C.map(i=>`
                            <div class="flex items-center gap-3">
                                <div class="w-16 h-16 bg-gray-100 dark:bg-stone-700 rounded-lg overflow-hidden flex-shrink-0">
                                    ${i.product?.image?`<img src="${i.product.image}" alt="" class="w-full h-full object-cover" onerror="this.parentElement.innerHTML='<div class=\\'w-full h-full flex items-center justify-center text-gray-400 dark:text-stone-500\\'><svg class=\\'w-6 h-6\\' fill=\\'none\\' stroke=\\'currentColor\\' viewBox=\\'0 0 24 24\\'><path stroke-linecap=\\'round\\' stroke-linejoin=\\'round\\' stroke-width=\\'1.5\\' d=\\'M4 16l4.586-4.586a2 2 0 012.828 0L16 16m-2-2l1.586-1.586a2 2 0 012.828 0L20 14m-6-6h.01M6 20h12a2 2 0 002-2V6a2 2 0 00-2-2H6a2 2 0 00-2 2v12a2 2 0 002 2z\\'/></svg></div>'">`:'<div class="w-full h-full flex items-center justify-center text-gray-400 dark:text-stone-500"><svg class="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="1.5" d="M4 16l4.586-4.586a2 2 0 012.828 0L16 16m-2-2l1.586-1.586a2 2 0 012.828 0L20 14m-6-6h.01M6 20h12a2 2 0 002-2V6a2 2 0 00-2-2H6a2 2 0 00-2 2v12a2 2 0 002 2z"/></svg></div>'}
                                </div>
                                <div>
                                    <p class="text-sm font-medium text-gray-900 dark:text-white">${Templates.escapeHtml(i.product?.name||i.product_name)}</p>
                                    <p class="text-sm text-gray-500 dark:text-stone-400">Qty: ${i.quantity}</p>
                                </div>
                            </div>
                        `).join("")}
                        ${k>0?`
                            <div class="flex items-center justify-center w-16 h-16 bg-gray-100 dark:bg-stone-700 rounded-lg">
                                <span class="text-sm text-gray-500 dark:text-stone-400">+${k}</span>
                            </div>
                        `:""}
                    </div>
                    <div class="mt-4 pt-4 border-t border-gray-100 flex justify-between items-center">
                        <p class="text-sm text-gray-600">
                            ${B.length} ${B.length===1?"item":"items"}
                        </p>
                        <p class="font-semibold text-gray-900">Total: ${Templates.formatPrice(E.total)}</p>
                    </div>
                </div>
            </div>
        `}async function L(E){let H=document.getElementById("order-detail");if(H){Loader.show(H,"skeleton");try{let B=(await OrdersApi.getById(E)).data;if(!B){window.location.href="/orders/";return}A(B)}catch(U){console.error("Failed to load order:",U),H.innerHTML='<p class="text-red-500 text-center py-8">Failed to load order details.</p>'}}}function A(E){let H=document.getElementById("order-detail");if(!H)return;let B={pending:"bg-yellow-100 text-yellow-700",processing:"bg-blue-100 text-blue-700",shipped:"bg-indigo-100 text-indigo-700",delivered:"bg-green-100 text-green-700",cancelled:"bg-red-100 text-red-700",refunded:"bg-gray-100 text-gray-700"}[E.status]||"bg-gray-100 text-gray-700",C=E.items||[];H.innerHTML=`
            <div class="mb-6">
                <a href="/orders/" class="inline-flex items-center text-primary-600 hover:text-primary-700">
                    <svg class="w-5 h-5 mr-2" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M7 16l-4-4m0 0l4-4m-4 4h18"/>
                    </svg>
                    Back to Orders
                </a>
            </div>

            <div class="bg-white rounded-xl shadow-sm border border-gray-100 overflow-hidden">
                <div class="px-6 py-4 border-b border-gray-100">
                    <div class="flex flex-wrap items-center justify-between gap-4">
                        <div>
                            <h1 class="text-xl font-bold text-gray-900">Order #${Templates.escapeHtml(E.order_number||E.id)}</h1>
                            <p class="text-sm text-gray-500">Placed on ${Templates.formatDate(E.created_at)}</p>
                        </div>
                        <span class="px-4 py-1.5 rounded-full text-sm font-medium ${B}">
                            ${Templates.escapeHtml(E.status_display||E.status)}
                        </span>
                    </div>
                </div>

                <!-- Order Timeline -->
                ${E.timeline&&E.timeline.length>0?`
                    <div class="px-6 py-4 border-b border-gray-100">
                        <h2 class="text-sm font-semibold text-gray-900 mb-4">Order Timeline</h2>
                        <div class="relative">
                            <div class="absolute left-2 top-2 bottom-2 w-0.5 bg-gray-200"></div>
                            <div class="space-y-4">
                                ${E.timeline.map((k,S)=>`
                                    <div class="relative pl-8">
                                        <div class="absolute left-0 w-4 h-4 rounded-full ${S===0?"bg-primary-600":"bg-gray-300"}"></div>
                                        <p class="text-sm font-medium text-gray-900">${Templates.escapeHtml(k.status)}</p>
                                        <p class="text-xs text-gray-500">${Templates.formatDate(k.timestamp,{includeTime:!0})}</p>
                                        ${k.note?`<p class="text-sm text-gray-600 mt-1">${Templates.escapeHtml(k.note)}</p>`:""}
                                    </div>
                                `).join("")}
                            </div>
                        </div>
                    </div>
                `:""}

                <!-- Tracking Info -->
                ${E.tracking_number?`
                    <div class="px-6 py-4 border-b border-gray-100 bg-blue-50">
                        <div class="flex items-center justify-between">
                            <div>
                                <p class="text-sm font-medium text-blue-900">Tracking Number</p>
                                <p class="text-lg font-mono text-blue-700">${Templates.escapeHtml(E.tracking_number)}</p>
                            </div>
                            ${E.tracking_url?`
                                <a href="${E.tracking_url}" target="_blank" class="px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-colors text-sm font-medium">
                                    Track Package
                                </a>
                            `:""}
                        </div>
                    </div>
                `:""}

                <!-- Order Items -->
                <div class="px-6 py-4 border-b border-gray-100">
                    <h2 class="text-sm font-semibold text-gray-900 mb-4">Items Ordered</h2>
                    <div class="space-y-4">
                        ${C.map(k=>`
                            <div class="flex gap-4">
                                <div class="w-20 h-20 bg-gray-100 rounded-lg overflow-hidden flex-shrink-0">
                                    ${k.product?.image?`<img src="${k.product.image}" alt="" class="w-full h-full object-cover" onerror="this.style.display='none'">`:""}
                                </div>
                                <div class="flex-1">
                                    <div class="flex justify-between">
                                        <div>
                                            <h3 class="font-medium text-gray-900">${Templates.escapeHtml(k.product?.name||k.product_name)}</h3>
                                            ${k.variant?`<p class="text-sm text-gray-500">${Templates.escapeHtml(k.variant.name||k.variant_name)}</p>`:""}
                                            <p class="text-sm text-gray-500">Qty: ${k.quantity}</p>
                                        </div>
                                        <p class="font-medium text-gray-900">${Templates.formatPrice(k.price*k.quantity)}</p>
                                    </div>
                                    ${k.product?.slug?`
                                        <a href="/products/${k.product.slug}/" class="text-sm text-primary-600 hover:text-primary-700 mt-2 inline-block">
                                            View Product
                                        </a>
                                    `:""}
                                </div>
                            </div>
                        `).join("")}
                    </div>
                </div>

                <!-- Addresses -->
                <div class="px-6 py-4 border-b border-gray-100 grid grid-cols-1 md:grid-cols-2 gap-6">
                    <div>
                        <h2 class="text-sm font-semibold text-gray-900 mb-2">Shipping Address</h2>
                        ${E.shipping_address?`
                            <p class="text-sm text-gray-600">${Templates.escapeHtml(E.shipping_address.full_name||`${E.shipping_address.first_name} ${E.shipping_address.last_name}`)}</p>
                            <p class="text-sm text-gray-600">${Templates.escapeHtml(E.shipping_address.address_line_1)}</p>
                            ${E.shipping_address.address_line_2?`<p class="text-sm text-gray-600">${Templates.escapeHtml(E.shipping_address.address_line_2)}</p>`:""}
                            <p class="text-sm text-gray-600">${Templates.escapeHtml(E.shipping_address.city)}, ${Templates.escapeHtml(E.shipping_address.state||"")} ${Templates.escapeHtml(E.shipping_address.postal_code)}</p>
                            <p class="text-sm text-gray-600">${Templates.escapeHtml(E.shipping_address.country)}</p>
                        `:'<p class="text-sm text-gray-500">Not available</p>'}
                    </div>
                    <div>
                        <h2 class="text-sm font-semibold text-gray-900 mb-2">Billing Address</h2>
                        ${E.billing_address?`
                            <p class="text-sm text-gray-600">${Templates.escapeHtml(E.billing_address.full_name||`${E.billing_address.first_name} ${E.billing_address.last_name}`)}</p>
                            <p class="text-sm text-gray-600">${Templates.escapeHtml(E.billing_address.address_line_1)}</p>
                            ${E.billing_address.address_line_2?`<p class="text-sm text-gray-600">${Templates.escapeHtml(E.billing_address.address_line_2)}</p>`:""}
                            <p class="text-sm text-gray-600">${Templates.escapeHtml(E.billing_address.city)}, ${Templates.escapeHtml(E.billing_address.state||"")} ${Templates.escapeHtml(E.billing_address.postal_code)}</p>
                            <p class="text-sm text-gray-600">${Templates.escapeHtml(E.billing_address.country)}</p>
                        `:'<p class="text-sm text-gray-500">Same as shipping</p>'}
                    </div>
                </div>

                <!-- Order Summary -->
                <div class="px-6 py-4">
                    <h2 class="text-sm font-semibold text-gray-900 mb-4">Order Summary</h2>
                    <div class="space-y-2 text-sm">
                        <div class="flex justify-between">
                            <span class="text-gray-600">Subtotal</span>
                            <span class="font-medium">${Templates.formatPrice(E.subtotal||0)}</span>
                        </div>
                        ${E.discount_amount?`
                            <div class="flex justify-between text-green-600">
                                <span>Discount</span>
                                <span>-${Templates.formatPrice(E.discount_amount)}</span>
                            </div>
                        `:""}
                        <div class="flex justify-between">
                            <span class="text-gray-600">Shipping</span>
                            <span class="font-medium">${E.shipping_cost>0?Templates.formatPrice(E.shipping_cost):"Free"}</span>
                        </div>
                        ${E.tax_amount?`
                            <div class="flex justify-between">
                                <span class="text-gray-600">Tax</span>
                                <span class="font-medium">${Templates.formatPrice(E.tax_amount)}</span>
                            </div>
                        `:""}
                        <div class="flex justify-between pt-2 border-t border-gray-200">
                            <span class="font-semibold text-gray-900">Total</span>
                            <span class="font-bold text-gray-900">${Templates.formatPrice(E.total)}</span>
                        </div>
                    </div>
                </div>
            </div>

            <!-- Actions -->
            <div class="mt-6 flex flex-wrap gap-4">
                ${E.status==="delivered"?`
                    <button id="reorder-btn" class="px-6 py-3 bg-primary-600 text-white font-semibold rounded-lg hover:bg-primary-700 transition-colors" data-order-id="${E.id}">
                        Order Again
                    </button>
                `:""}
                ${["pending","processing"].includes(E.status)?`
                    <button id="cancel-order-btn" class="px-6 py-3 border border-red-300 text-red-600 font-semibold rounded-lg hover:bg-red-50 transition-colors" data-order-id="${E.id}">
                        Cancel Order
                    </button>
                `:""}
                <button id="print-invoice-btn" class="px-6 py-3 border border-gray-300 text-gray-700 font-semibold rounded-lg hover:bg-gray-50 transition-colors">
                    Print Invoice
                </button>
            </div>
        `,Y(E)}function Y(E){let H=document.getElementById("reorder-btn"),U=document.getElementById("cancel-order-btn"),B=document.getElementById("print-invoice-btn");H?.addEventListener("click",async()=>{try{await OrdersApi.reorder(E.id),Toast.success("Items added to cart!"),document.dispatchEvent(new CustomEvent("cart:updated")),window.location.href="/cart/"}catch(C){Toast.error(C.message||"Failed to reorder.")}}),U?.addEventListener("click",async()=>{if(await Modal.confirm({title:"Cancel Order",message:"Are you sure you want to cancel this order? This action cannot be undone.",confirmText:"Cancel Order",cancelText:"Keep Order"}))try{await OrdersApi.cancel(E.id),Toast.success("Order cancelled."),window.location.reload()}catch(k){Toast.error(k.message||"Failed to cancel order.")}}),B?.addEventListener("click",()=>{window.print()})}function Z(){let E=document.querySelectorAll("[data-filter-status]");E.forEach(H=>{H.addEventListener("click",()=>{E.forEach(U=>{U.classList.remove("bg-primary-100","text-primary-700"),U.classList.add("text-gray-600")}),H.classList.add("bg-primary-100","text-primary-700"),H.classList.remove("text-gray-600"),e=H.dataset.filterStatus,n=1,h()})})}function z(){n=1,e="all"}return{init:s,destroy:z}})();window.OrdersPage=mr;Ur=mr});var hr=_t(()=>{var Wr=(function(){"use strict";let n=window.BUNORAA_ROUTES||{},e=n.preordersWizard||"/preorders/create/",s=n.preordersLanding||"/preorders/";async function o(){await Promise.all([h(),_(),L()])}async function h(){let B=document.getElementById("preorder-categories");if(B)try{let C=await PreordersApi.getCategories({featured:!0,pageSize:8}),k=C?.data?.results||C?.data||C?.results||[];if(k.length===0){B.innerHTML=`
                    <div class="col-span-full text-center py-12">
                        <svg class="w-16 h-16 mx-auto text-gray-400 mb-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="1.5" 
                                  d="M20 7l-8-4-8 4m16 0l-8 4m8-4v10l-8 4m0-10L4 7m8 4v10M4 7v10l8 4"/>
                        </svg>
                        <p class="text-gray-600 dark:text-gray-400 mb-4">No categories available at the moment</p>
                        <p class="text-sm text-gray-500 dark:text-gray-500">Check back soon or contact us for custom requests</p>
                    </div>
                `;return}B.innerHTML=k.map(S=>w(S)).join("")}catch(C){console.error("Failed to load pre-order categories:",C),B.innerHTML=`
                <div class="col-span-full text-center py-12">
                    <svg class="w-16 h-16 mx-auto text-gray-400 mb-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path stroke-linecap="round" stroke-linejoin="round" stroke-width="1.5" 
                              d="M20 7l-8-4-8 4m16 0l-8 4m8-4v10l-8 4m0-10L4 7m8 4v10M4 7v10l8 4"/>
                    </svg>
                    <p class="text-gray-600 dark:text-gray-400">No categories available at the moment</p>
                </div>
            `}}function w(B){let C=B.image?.url||B.image||B.thumbnail||"",k=C&&C.length>0,S=Templates?.escapeHtml||(y=>y),T=Templates?.formatPrice||(y=>`${window.BUNORAA_CURRENCY?.symbol||"\u09F3"}${y}`);return`
            <a href="${s}category/${B.slug}/" 
               class="group bg-white dark:bg-gray-800 rounded-2xl shadow-sm hover:shadow-xl transition-all duration-300 overflow-hidden border border-gray-200 dark:border-gray-700">
                ${k?`
                    <div class="aspect-video relative overflow-hidden">
                        <img src="${C}" alt="${S(B.name)}" 
                             class="w-full h-full object-cover group-hover:scale-105 transition-transform duration-300"
                             loading="lazy">
                        <div class="absolute inset-0 bg-gradient-to-t from-black/50 to-transparent"></div>
                    </div>
                `:`
                    <div class="aspect-video bg-gradient-to-br from-purple-500 to-indigo-600 flex items-center justify-center">
                        ${B.icon?`<span class="text-5xl">${B.icon}</span>`:`
                            <svg class="w-16 h-16 text-white/80" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="1.5" 
                                      d="M20 7l-8-4-8 4m16 0l-8 4m8-4v10l-8 4m0-10L4 7m8 4v10M4 7v10l8 4"/>
                            </svg>
                        `}
                    </div>
                `}
                
                <div class="p-6">
                    <div class="flex items-start justify-between mb-3">
                        <h3 class="text-xl font-bold text-gray-900 dark:text-white group-hover:text-purple-600 transition-colors">
                            ${S(B.name)}
                        </h3>
                        ${B.icon?`<span class="text-2xl">${B.icon}</span>`:""}
                    </div>
                    
                    ${B.description?`
                        <p class="text-gray-600 dark:text-gray-400 mb-4 line-clamp-2">
                            ${S(B.description)}
                        </p>
                    `:""}
                    
                    <div class="flex items-center justify-between text-sm">
                        ${B.base_price?`
                            <span class="text-gray-500 dark:text-gray-500">
                                Starting from <span class="font-semibold text-purple-600">${T(B.base_price)}</span>
                            </span>
                        `:"<span></span>"}
                        ${B.min_production_days&&B.max_production_days?`
                            <span class="text-gray-500 dark:text-gray-500">
                                ${B.min_production_days}-${B.max_production_days} days
                            </span>
                        `:""}
                    </div>
                    
                    <div class="mt-4 pt-4 border-t border-gray-100 dark:border-gray-700 flex items-center justify-between">
                        <div class="flex gap-2 flex-wrap">
                            ${B.requires_design?`
                                <span class="text-xs px-2 py-1 bg-blue-100 dark:bg-blue-900/30 text-blue-700 dark:text-blue-300 rounded-full">
                                    Design Required
                                </span>
                            `:""}
                            ${B.allow_rush_order?`
                                <span class="text-xs px-2 py-1 bg-orange-100 dark:bg-orange-900/30 text-orange-700 dark:text-orange-300 rounded-full">
                                    Rush Available
                                </span>
                            `:""}
                        </div>
                        <svg class="w-5 h-5 text-gray-400 group-hover:text-purple-600 group-hover:translate-x-1 transition-all flex-shrink-0" 
                             fill="none" stroke="currentColor" viewBox="0 0 24 24">
                            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9 5l7 7-7 7"/>
                        </svg>
                    </div>
                </div>
            </a>
        `}async function _(){let B=document.getElementById("preorder-templates");B&&B.closest("section")?.classList.add("hidden")}async function L(){let B=document.getElementById("preorder-stats");if(!B)return;let C={totalOrders:"500+",happyCustomers:"450+",avgRating:"4.9"};B.innerHTML=`
            <div class="flex items-center gap-8 justify-center flex-wrap">
                <div class="text-center">
                    <p class="text-3xl font-bold text-purple-600 dark:text-purple-400">${C.totalOrders}</p>
                    <p class="text-sm text-gray-600 dark:text-gray-400">Orders Completed</p>
                </div>
                <div class="h-12 w-px bg-gray-200 dark:bg-gray-700 hidden sm:block"></div>
                <div class="text-center">
                    <p class="text-3xl font-bold text-purple-600 dark:text-purple-400">${C.happyCustomers}</p>
                    <p class="text-sm text-gray-600 dark:text-gray-400">Happy Customers</p>
                </div>
                <div class="h-12 w-px bg-gray-200 dark:bg-gray-700 hidden sm:block"></div>
                <div class="text-center">
                    <p class="text-3xl font-bold text-purple-600 dark:text-purple-400">${C.avgRating}</p>
                    <p class="text-sm text-gray-600 dark:text-gray-400">Average Rating</p>
                </div>
            </div>
        `}async function A(B){let C=document.getElementById("category-options");if(!(!C||!B))try{let k=await PreordersApi.getCategory(B),S=await PreordersApi.getCategoryOptions(k.id);Y(C,S)}catch(k){console.error("Failed to load category options:",k)}}function Y(B,C){if(!C||C.length===0){B.innerHTML='<p class="text-gray-500">No customization options available.</p>';return}B.innerHTML=C.map(k=>`
            <div class="border border-gray-200 dark:border-gray-700 rounded-xl p-4">
                <h4 class="font-semibold text-gray-900 dark:text-white mb-2">${Templates.escapeHtml(k.name)}</h4>
                ${k.description?`<p class="text-sm text-gray-600 dark:text-gray-400 mb-3">${Templates.escapeHtml(k.description)}</p>`:""}
                <div class="space-y-2">
                    ${k.choices?.map(S=>`
                        <label class="flex items-center gap-3 p-3 border border-gray-200 dark:border-gray-600 rounded-lg cursor-pointer hover:bg-gray-50 dark:hover:bg-gray-800 transition-colors">
                            <input type="${k.allow_multiple?"checkbox":"radio"}" name="option_${k.id}" value="${S.id}" class="text-purple-600 focus:ring-purple-500">
                            <span class="flex-1">
                                <span class="font-medium text-gray-900 dark:text-white">${Templates.escapeHtml(S.name)}</span>
                                ${S.price_modifier&&S.price_modifier!=="0.00"?`
                                    <span class="text-sm text-purple-600 dark:text-purple-400 ml-2">+${Templates.formatPrice(S.price_modifier)}</span>
                                `:""}
                            </span>
                        </label>
                    `).join("")||""}
                </div>
            </div>
        `).join("")}async function Z(){let B=document.getElementById("my-preorders-list");if(B){Loader.show(B,"skeleton");try{let C=await PreordersApi.getMyPreorders(),k=C?.data?.results||C?.data||C?.results||[];if(k.length===0){B.innerHTML=`
                    <div class="text-center py-12">
                        <svg class="w-20 h-20 text-gray-300 mx-auto mb-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z"/>
                        </svg>
                        <h3 class="text-xl font-semibold text-gray-900 dark:text-white mb-2">No custom orders yet</h3>
                        <p class="text-gray-600 dark:text-gray-400 mb-6">Start creating your first custom order today!</p>
                        <a href="${e}" class="inline-flex items-center gap-2 px-6 py-3 bg-purple-600 text-white font-semibold rounded-xl hover:bg-purple-700 transition-colors">
                            Create Custom Order
                            <svg class="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 4v16m8-8H4"/></svg>
                        </a>
                    </div>
                `;return}B.innerHTML=k.map(S=>z(S)).join("")}catch(C){console.error("Failed to load pre-orders:",C),B.innerHTML=`
                <div class="text-center py-12">
                    <p class="text-red-500">Failed to load your orders. Please try again.</p>
                </div>
            `}}}function z(B){let C={pending:"bg-yellow-100 text-yellow-800 dark:bg-yellow-900/30 dark:text-yellow-400",quoted:"bg-blue-100 text-blue-800 dark:bg-blue-900/30 dark:text-blue-400",accepted:"bg-indigo-100 text-indigo-800 dark:bg-indigo-900/30 dark:text-indigo-400",in_progress:"bg-purple-100 text-purple-800 dark:bg-purple-900/30 dark:text-purple-400",review:"bg-orange-100 text-orange-800 dark:bg-orange-900/30 dark:text-orange-400",approved:"bg-teal-100 text-teal-800 dark:bg-teal-900/30 dark:text-teal-400",completed:"bg-green-100 text-green-800 dark:bg-green-900/30 dark:text-green-400",cancelled:"bg-red-100 text-red-800 dark:bg-red-900/30 dark:text-red-400",refunded:"bg-gray-100 text-gray-800 dark:bg-gray-900/30 dark:text-gray-400"},k={pending:"Pending Review",quoted:"Quote Sent",accepted:"Quote Accepted",in_progress:"In Progress",review:"Under Review",approved:"Approved",completed:"Completed",cancelled:"Cancelled",refunded:"Refunded"},S=C[B.status]||"bg-gray-100 text-gray-800",T=k[B.status]||B.status;return`
            <a href="${s}order/${B.preorder_number}/" class="block bg-white dark:bg-gray-800 rounded-2xl border border-gray-200 dark:border-gray-700 p-6 hover:shadow-lg transition-shadow">
                <div class="flex items-start justify-between gap-4 mb-4">
                    <div>
                        <p class="text-sm text-gray-500 dark:text-gray-400">${B.preorder_number}</p>
                        <h3 class="text-lg font-semibold text-gray-900 dark:text-white">${Templates.escapeHtml(B.title||B.category?.name||"Custom Order")}</h3>
                    </div>
                    <span class="px-3 py-1 text-xs font-medium rounded-full ${S}">${T}</span>
                </div>
                ${B.description?`
                    <p class="text-gray-600 dark:text-gray-400 text-sm mb-4 line-clamp-2">${Templates.escapeHtml(B.description)}</p>
                `:""}
                <div class="flex items-center justify-between text-sm">
                    <span class="text-gray-500 dark:text-gray-400">
                        ${new Date(B.created_at).toLocaleDateString()}
                    </span>
                    ${B.final_price||B.estimated_price?`
                        <span class="font-semibold text-purple-600 dark:text-purple-400">
                            ${Templates.formatPrice(B.final_price||B.estimated_price)}
                        </span>
                    `:""}
                </div>
            </a>
        `}async function E(B){B&&await Promise.all([H(B),U(B)])}async function H(B){if(document.getElementById("preorder-status"))try{let k=await PreordersApi.getPreorderStatus(B)}catch{}}function U(B){let C=document.getElementById("message-form");C&&C.addEventListener("submit",async k=>{k.preventDefault();let S=C.querySelector('textarea[name="message"]'),T=C.querySelector('button[type="submit"]'),y=S?.value?.trim();if(!y){Toast.error("Please enter a message");return}let i=T.innerHTML;T.disabled=!0,T.innerHTML='<svg class="animate-spin h-5 w-5 mx-auto" fill="none" viewBox="0 0 24 24"><circle class="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" stroke-width="4"></circle><path class="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path></svg>';try{await PreordersApi.sendMessage(B,y),Toast.success("Message sent successfully"),S.value="",location.reload()}catch(a){Toast.error(a.message||"Failed to send message")}finally{T.disabled=!1,T.innerHTML=i}})}return{initLanding:o,initCategoryDetail:A,initMyPreorders:Z,initDetail:E,loadFeaturedCategories:h,renderCategoryCard:w,renderPreorderCard:z}})();window.PreordersPage=Wr});var yr={};te(yr,{default:()=>Yr});var fr,Yr,br=ee(()=>{fr=(function(){"use strict";let n=null,e=null,s=null,o=!1,h=!1,w=!1,_=null;async function L(){if(o)return;o=!0;let t=document.getElementById("product-detail");if(!t)return;let r=t.querySelector("h1")||t.dataset.productId;if(h=!!r,r){n={id:t.dataset.productId,slug:t.dataset.productSlug},S(),A();return}let d=u();if(!d){window.location.href="/products/";return}await m(d),A()}function A(){Y(),Z(),z(),E(),H(),U(),B(),C(),k()}function Y(){let t=document.getElementById("main-product-image")||document.getElementById("main-image"),r=t?.parentElement;if(!t||!r)return;let d=document.createElement("div");d.className="zoom-lens absolute w-32 h-32 border-2 border-primary-500 bg-white/30 pointer-events-none opacity-0 transition-opacity duration-200 z-10",d.style.backgroundRepeat="no-repeat";let l=document.createElement("div");l.className="zoom-result fixed right-8 top-1/2 -translate-y-1/2 w-96 h-96 border border-stone-200 dark:border-stone-700 rounded-xl shadow-2xl bg-white dark:bg-stone-800 opacity-0 pointer-events-none transition-opacity duration-200 z-50 hidden lg:block",l.style.backgroundRepeat="no-repeat",r.classList.add("relative"),r.appendChild(d),document.body.appendChild(l),r.addEventListener("mouseenter",()=>{window.innerWidth<1024||(d.classList.remove("opacity-0"),l.classList.remove("opacity-0"),w=!0)}),r.addEventListener("mouseleave",()=>{d.classList.add("opacity-0"),l.classList.add("opacity-0"),w=!1}),r.addEventListener("mousemove",p=>{if(!w||window.innerWidth<1024)return;let f=r.getBoundingClientRect(),I=p.clientX-f.left,F=p.clientY-f.top,Q=I-d.offsetWidth/2,K=F-d.offsetHeight/2;d.style.left=`${Math.max(0,Math.min(f.width-d.offsetWidth,Q))}px`,d.style.top=`${Math.max(0,Math.min(f.height-d.offsetHeight,K))}px`;let ae=3,ge=-I*ae+l.offsetWidth/2,Te=-F*ae+l.offsetHeight/2;l.style.backgroundImage=`url(${t.src})`,l.style.backgroundSize=`${f.width*ae}px ${f.height*ae}px`,l.style.backgroundPosition=`${ge}px ${Te}px`})}function Z(){let t=document.getElementById("size-guide-btn");t&&t.addEventListener("click",()=>{let r=document.createElement("div");r.id="size-guide-modal",r.className="fixed inset-0 z-50 flex items-center justify-center p-4",r.innerHTML=`
                <div class="absolute inset-0 bg-black/50 backdrop-blur-sm" onclick="document.getElementById('size-guide-modal').remove()"></div>
                <div class="relative bg-white dark:bg-stone-800 rounded-2xl shadow-2xl max-w-2xl w-full max-h-[90vh] overflow-auto">
                    <div class="sticky top-0 bg-white dark:bg-stone-800 border-b border-stone-200 dark:border-stone-700 p-4 flex items-center justify-between">
                        <h2 class="text-xl font-bold text-stone-900 dark:text-white">Size Guide</h2>
                        <button onclick="document.getElementById('size-guide-modal').remove()" class="w-10 h-10 bg-stone-100 dark:bg-stone-700 rounded-full flex items-center justify-center hover:bg-stone-200 dark:hover:bg-stone-600 transition-colors">
                            <svg class="w-5 h-5 text-stone-600 dark:text-stone-300" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M6 18L18 6M6 6l12 12"/></svg>
                        </button>
                    </div>
                    <div class="p-6">
                        <div class="mb-6">
                            <h3 class="text-lg font-semibold text-stone-900 dark:text-white mb-2">How to Measure</h3>
                            <p class="text-stone-600 dark:text-stone-400 text-sm">Use a flexible measuring tape for accurate measurements. Measure over your undergarments for best results.</p>
                        </div>
                        <div class="overflow-x-auto">
                            <table class="w-full text-sm">
                                <thead>
                                    <tr class="bg-stone-50 dark:bg-stone-700">
                                        <th class="px-4 py-3 text-left font-semibold text-stone-900 dark:text-white">Size</th>
                                        <th class="px-4 py-3 text-center font-semibold text-stone-900 dark:text-white">Chest (in)</th>
                                        <th class="px-4 py-3 text-center font-semibold text-stone-900 dark:text-white">Waist (in)</th>
                                        <th class="px-4 py-3 text-center font-semibold text-stone-900 dark:text-white">Length (in)</th>
                                    </tr>
                                </thead>
                                <tbody>
                                    <tr class="border-b border-stone-100 dark:border-stone-600">
                                        <td class="px-4 py-3 font-medium text-stone-900 dark:text-white">XS</td>
                                        <td class="px-4 py-3 text-center text-stone-600 dark:text-stone-400">32-34</td>
                                        <td class="px-4 py-3 text-center text-stone-600 dark:text-stone-400">26-28</td>
                                        <td class="px-4 py-3 text-center text-stone-600 dark:text-stone-400">26</td>
                                    </tr>
                                    <tr class="border-b border-stone-100 dark:border-stone-600">
                                        <td class="px-4 py-3 font-medium text-stone-900 dark:text-white">S</td>
                                        <td class="px-4 py-3 text-center text-stone-600 dark:text-stone-400">35-37</td>
                                        <td class="px-4 py-3 text-center text-stone-600 dark:text-stone-400">29-31</td>
                                        <td class="px-4 py-3 text-center text-stone-600 dark:text-stone-400">27</td>
                                    </tr>
                                    <tr class="border-b border-stone-100 dark:border-stone-600">
                                        <td class="px-4 py-3 font-medium text-stone-900 dark:text-white">M</td>
                                        <td class="px-4 py-3 text-center text-stone-600 dark:text-stone-400">38-40</td>
                                        <td class="px-4 py-3 text-center text-stone-600 dark:text-stone-400">32-34</td>
                                        <td class="px-4 py-3 text-center text-stone-600 dark:text-stone-400">28</td>
                                    </tr>
                                    <tr class="border-b border-stone-100 dark:border-stone-600">
                                        <td class="px-4 py-3 font-medium text-stone-900 dark:text-white">L</td>
                                        <td class="px-4 py-3 text-center text-stone-600 dark:text-stone-400">41-43</td>
                                        <td class="px-4 py-3 text-center text-stone-600 dark:text-stone-400">35-37</td>
                                        <td class="px-4 py-3 text-center text-stone-600 dark:text-stone-400">29</td>
                                    </tr>
                                    <tr>
                                        <td class="px-4 py-3 font-medium text-stone-900 dark:text-white">XL</td>
                                        <td class="px-4 py-3 text-center text-stone-600 dark:text-stone-400">44-46</td>
                                        <td class="px-4 py-3 text-center text-stone-600 dark:text-stone-400">38-40</td>
                                        <td class="px-4 py-3 text-center text-stone-600 dark:text-stone-400">30</td>
                                    </tr>
                                </tbody>
                            </table>
                        </div>
                        <div class="mt-6 p-4 bg-amber-50 dark:bg-amber-900/20 rounded-xl">
                            <p class="text-sm text-amber-800 dark:text-amber-200">\u{1F4A1} <strong>Tip:</strong> If you're between sizes, we recommend sizing up for a more comfortable fit.</p>
                        </div>
                    </div>
                </div>
            `,document.body.appendChild(r)})}function z(){let t=document.getElementById("stock-alert-btn");t&&t.addEventListener("click",()=>{if(!document.getElementById("product-detail")?.dataset.productId)return;let d=document.createElement("div");d.id="stock-alert-modal",d.className="fixed inset-0 z-50 flex items-center justify-center p-4",d.innerHTML=`
                <div class="absolute inset-0 bg-black/50 backdrop-blur-sm" onclick="document.getElementById('stock-alert-modal').remove()"></div>
                <div class="relative bg-white dark:bg-stone-800 rounded-2xl shadow-2xl max-w-md w-full p-6">
                    <button onclick="document.getElementById('stock-alert-modal').remove()" class="absolute top-4 right-4 w-8 h-8 bg-stone-100 dark:bg-stone-700 rounded-full flex items-center justify-center hover:bg-stone-200 dark:hover:bg-stone-600 transition-colors">
                        <svg class="w-4 h-4 text-stone-600 dark:text-stone-300" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M6 18L18 6M6 6l12 12"/></svg>
                    </button>
                    <div class="text-center mb-6">
                        <div class="w-16 h-16 bg-primary-100 dark:bg-amber-900/30 rounded-full flex items-center justify-center mx-auto mb-4">
                            <svg class="w-8 h-8 text-primary-600 dark:text-amber-400" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M15 17h5l-1.405-1.405A2.032 2.032 0 0118 14.158V11a6.002 6.002 0 00-4-5.659V5a2 2 0 10-4 0v.341C7.67 6.165 6 8.388 6 11v3.159c0 .538-.214 1.055-.595 1.436L4 17h5m6 0v1a3 3 0 11-6 0v-1m6 0H9"/></svg>
                        </div>
                        <h3 class="text-xl font-bold text-stone-900 dark:text-white mb-2">Notify Me When Available</h3>
                        <p class="text-stone-600 dark:text-stone-400 text-sm">We'll email you when this product is back in stock.</p>
                    </div>
                    <form id="stock-alert-form" class="space-y-4">
                        <input type="email" id="stock-alert-email" placeholder="Enter your email" required class="w-full px-4 py-3 border border-stone-300 dark:border-stone-600 rounded-xl bg-white dark:bg-stone-700 text-stone-900 dark:text-white focus:ring-2 focus:ring-primary-500 dark:focus:ring-amber-500">
                        <button type="submit" class="w-full py-3 bg-primary-600 dark:bg-amber-600 text-white font-semibold rounded-xl hover:bg-primary-700 dark:hover:bg-amber-700 transition-colors">
                            Notify Me
                        </button>
                    </form>
                </div>
            `,document.body.appendChild(d),document.getElementById("stock-alert-form")?.addEventListener("submit",async l=>{l.preventDefault();let p=document.getElementById("stock-alert-email")?.value;if(p)try{_=p,Toast.success("You will be notified when this product is back in stock!"),d.remove()}catch{Toast.error("Failed to subscribe. Please try again.")}})})}function E(){document.querySelectorAll(".share-btn").forEach(t=>{t.addEventListener("click",()=>{let r=t.dataset.platform,d=encodeURIComponent(window.location.href),l=encodeURIComponent(document.title),p=document.querySelector("h1")?.textContent||"",f={facebook:`https://www.facebook.com/sharer/sharer.php?u=${d}`,twitter:`https://twitter.com/intent/tweet?url=${d}&text=${encodeURIComponent(p)}`,pinterest:`https://pinterest.com/pin/create/button/?url=${d}&description=${encodeURIComponent(p)}`,whatsapp:`https://api.whatsapp.com/send?text=${encodeURIComponent(p+" "+window.location.href)}`,copy:null};r==="copy"?navigator.clipboard.writeText(window.location.href).then(()=>{Toast.success("Link copied to clipboard!")}).catch(()=>{Toast.error("Failed to copy link")}):f[r]&&window.open(f[r],"_blank","width=600,height=400")})})}function H(){let t=document.getElementById("qa-section");if(!t||!document.getElementById("product-detail")?.dataset.productId)return;let d=[{question:"Is this product machine washable?",answer:"Yes, we recommend washing on a gentle cycle with cold water.",askedBy:"John D.",date:"2 days ago"},{question:"What materials is this made from?",answer:"This product is crafted from 100% organic cotton sourced from sustainable farms.",askedBy:"Sarah M.",date:"1 week ago"}];t.innerHTML=`
            <div class="space-y-6">
                <div class="flex items-center justify-between">
                    <h3 class="text-lg font-semibold text-stone-900 dark:text-white">Questions & Answers</h3>
                    <button id="ask-question-btn" class="text-sm font-medium text-primary-600 dark:text-amber-400 hover:underline">Ask a Question</button>
                </div>
                <div id="qa-list" class="space-y-4">
                    ${d.map(l=>`
                        <div class="bg-stone-50 dark:bg-stone-700/50 rounded-xl p-4">
                            <div class="flex items-start gap-3 mb-2">
                                <span class="text-primary-600 dark:text-amber-400 font-bold">Q:</span>
                                <div>
                                    <p class="text-stone-900 dark:text-white font-medium">${Templates.escapeHtml(l.question)}</p>
                                    <p class="text-xs text-stone-500 dark:text-stone-400 mt-1">${l.askedBy} \u2022 ${l.date}</p>
                                </div>
                            </div>
                            ${l.answer?`
                                <div class="flex items-start gap-3 mt-3 pl-6">
                                    <span class="text-emerald-600 dark:text-emerald-400 font-bold">A:</span>
                                    <p class="text-stone-600 dark:text-stone-300">${Templates.escapeHtml(l.answer)}</p>
                                </div>
                            `:""}
                        </div>
                    `).join("")}
                </div>
            </div>
        `,document.getElementById("ask-question-btn")?.addEventListener("click",()=>{let l=document.createElement("div");l.id="ask-question-modal",l.className="fixed inset-0 z-50 flex items-center justify-center p-4",l.innerHTML=`
                <div class="absolute inset-0 bg-black/50 backdrop-blur-sm" onclick="document.getElementById('ask-question-modal').remove()"></div>
                <div class="relative bg-white dark:bg-stone-800 rounded-2xl shadow-2xl max-w-md w-full p-6">
                    <button onclick="document.getElementById('ask-question-modal').remove()" class="absolute top-4 right-4 w-8 h-8 bg-stone-100 dark:bg-stone-700 rounded-full flex items-center justify-center">
                        <svg class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M6 18L18 6M6 6l12 12"/></svg>
                    </button>
                    <h3 class="text-xl font-bold text-stone-900 dark:text-white mb-4">Ask a Question</h3>
                    <form id="question-form" class="space-y-4">
                        <textarea id="question-input" rows="4" placeholder="Type your question here..." required class="w-full px-4 py-3 border border-stone-300 dark:border-stone-600 rounded-xl bg-white dark:bg-stone-700 text-stone-900 dark:text-white resize-none focus:ring-2 focus:ring-primary-500"></textarea>
                        <button type="submit" class="w-full py-3 bg-primary-600 dark:bg-amber-600 text-white font-semibold rounded-xl hover:bg-primary-700 dark:hover:bg-amber-700 transition-colors">
                            Submit Question
                        </button>
                    </form>
                </div>
            `,document.body.appendChild(l),document.getElementById("question-form")?.addEventListener("submit",p=>{p.preventDefault(),Toast.success("Your question has been submitted!"),l.remove()})})}function U(){let t=document.getElementById("delivery-estimate");if(!t)return;let r=new Date,d=3,l=7,p=new Date(r.getTime()+d*24*60*60*1e3),f=new Date(r.getTime()+l*24*60*60*1e3),I=F=>F.toLocaleDateString("en-US",{weekday:"short",month:"short",day:"numeric"});t.innerHTML=`
            <div class="flex items-start gap-3 p-4 bg-emerald-50 dark:bg-emerald-900/20 rounded-xl">
                <svg class="w-5 h-5 text-emerald-600 dark:text-emerald-400 mt-0.5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M5 8h14M5 8a2 2 0 110-4h14a2 2 0 110 4M5 8v10a2 2 0 002 2h10a2 2 0 002-2V8m-9 4h4"/>
                </svg>
                <div>
                    <p class="text-sm font-medium text-emerald-700 dark:text-emerald-300">Estimated Delivery</p>
                    <p class="text-emerald-600 dark:text-emerald-400 font-semibold">${I(p)} - ${I(f)}</p>
                    <p class="text-xs text-emerald-600 dark:text-emerald-400 mt-1">Free shipping on orders over $50</p>
                </div>
            </div>
        `}function B(){let t=document.getElementById("product-detail")?.dataset.productId,r=document.getElementById("product-detail")?.dataset.productSlug,d=document.querySelector("h1")?.textContent,l=document.getElementById("main-product-image")?.src||document.getElementById("main-image")?.src,p=document.getElementById("product-price")?.textContent;if(!t)return;let f=JSON.parse(localStorage.getItem("recentlyViewed")||"[]"),I=f.findIndex(F=>F.id===t);I>-1&&f.splice(I,1),f.unshift({id:t,slug:r,name:d,image:l,price:p,viewedAt:new Date().toISOString()}),localStorage.setItem("recentlyViewed",JSON.stringify(f.slice(0,20)))}function C(){if(document.getElementById("mobile-sticky-atc")||document.getElementById("mobile-sticky-atc-js")||window.innerWidth>=1024)return;let r=n;if(!r)return;let d=document.createElement("div");d.id="mobile-sticky-atc-enhanced",d.className="fixed bottom-0 inset-x-0 z-40 lg:hidden bg-white dark:bg-stone-800 border-t border-stone-200 dark:border-stone-700 shadow-2xl p-3 transform translate-y-full transition-transform duration-300",d.innerHTML=`
            <div class="flex items-center gap-3">
                <div class="flex-1 min-w-0">
                    <p class="text-xs text-stone-500 dark:text-stone-400 truncate">${r.name||""}</p>
                    <p class="font-bold text-stone-900 dark:text-white">${r.sale_price?Templates.formatPrice(r.sale_price):Templates.formatPrice(r.price||0)}</p>
                </div>
                <button id="sticky-add-to-cart" class="px-6 py-3 bg-primary-600 dark:bg-amber-600 text-white font-semibold rounded-xl hover:bg-primary-700 dark:hover:bg-amber-700 transition-colors">
                    Add to Cart
                </button>
            </div>
        `,document.body.appendChild(d);let l=document.getElementById("add-to-cart-btn");l&&new IntersectionObserver(f=>{f.forEach(I=>{I.isIntersecting?d.classList.add("translate-y-full"):d.classList.remove("translate-y-full")})},{threshold:0}).observe(l),document.getElementById("sticky-add-to-cart")?.addEventListener("click",()=>{document.getElementById("add-to-cart-btn")?.click()})}function k(){document.querySelectorAll("[data-video-url]").forEach(r=>{r.addEventListener("click",()=>{let d=r.dataset.videoUrl;if(!d)return;let l=document.createElement("div");l.id="video-player-modal",l.className="fixed inset-0 z-50 flex items-center justify-center bg-black/90",l.innerHTML=`
                    <button onclick="document.getElementById('video-player-modal').remove()" class="absolute top-4 right-4 w-12 h-12 bg-white/20 rounded-full flex items-center justify-center hover:bg-white/30 transition-colors">
                        <svg class="w-6 h-6 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M6 18L18 6M6 6l12 12"/></svg>
                    </button>
                    <video controls autoplay class="max-w-full max-h-[90vh] rounded-xl">
                        <source src="${d}" type="video/mp4">
                        Your browser does not support video playback.
                    </video>
                `,document.body.appendChild(l)})})}function S(){J(),y(),i(),t();async function t(){let r=document.getElementById("add-to-wishlist-btn");if(!r)return;let d=document.getElementById("product-detail")?.dataset.productId;if(d&&!(typeof WishlistApi>"u"))try{let l=await WishlistApi.getWishlist({pageSize:100});l.success&&l.data?.items&&(l.data.items.some(f=>f.product_id===d||f.product===d)?(r.querySelector("svg")?.setAttribute("fill","currentColor"),r.classList.add("text-red-500")):(r.querySelector("svg")?.setAttribute("fill","none"),r.classList.remove("text-red-500")))}catch{}}a(),c(),we(),T()}function T(){let t=document.querySelectorAll(".tab-btn"),r=document.querySelectorAll(".tab-content");t.forEach(d=>{d.addEventListener("click",()=>{let l=d.dataset.tab;t.forEach(p=>{p.classList.remove("border-primary-500","text-primary-600"),p.classList.add("border-transparent","text-gray-500")}),d.classList.add("border-primary-500","text-primary-600"),d.classList.remove("border-transparent","text-gray-500"),r.forEach(p=>{p.id===`${l}-tab`?p.classList.remove("hidden"):p.classList.add("hidden")})})})}function y(){let t=document.getElementById("add-to-cart-btn");t&&t.addEventListener("click",async()=>{let r=document.getElementById("product-detail")?.dataset.productId,d=parseInt(document.getElementById("quantity")?.value)||1,p=!!document.querySelector('input[name="variant"]'),f=document.querySelector('input[name="variant"]:checked')?.value;if(!r)return;if(p&&!f){Toast.warning("Please select a variant before adding to cart.");return}t.disabled=!0;let I=t.innerHTML;t.innerHTML='<svg class="animate-spin h-5 w-5 mx-auto" fill="none" viewBox="0 0 24 24"><circle class="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" stroke-width="4"></circle><path class="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path></svg>';try{await CartApi.addItem(r,d,f||null),Toast.success("Added to cart!"),document.dispatchEvent(new CustomEvent("cart:updated"))}catch(F){Toast.error(F.message||"Failed to add to cart.")}finally{t.disabled=!1,t.innerHTML=I}})}function i(){let t=document.getElementById("add-to-wishlist-btn");t&&t.addEventListener("click",async()=>{let r=document.getElementById("product-detail")?.dataset.productId;if(r){if(typeof AuthApi<"u"&&!AuthApi.isAuthenticated()){Toast.warning("Please login to add items to your wishlist."),window.location.href="/account/login/?next="+encodeURIComponent(window.location.pathname);return}try{let d=!1;if(typeof WishlistApi<"u"){let l=await WishlistApi.getWishlist({pageSize:100});l.success&&l.data?.items&&(d=l.data.items.some(p=>p.product_id===r||p.product===r))}if(d){let p=(await WishlistApi.getWishlist({pageSize:100})).data.items.find(f=>f.product_id===r||f.product===r);p&&(await WishlistApi.removeItem(p.id),Toast.success("Removed from wishlist!"),t.querySelector("svg")?.setAttribute("fill","none"),t.classList.remove("text-red-500"),t.setAttribute("aria-pressed","false"))}else await WishlistApi.addItem(r),Toast.success("Added to wishlist!"),t.querySelector("svg")?.setAttribute("fill","currentColor"),t.classList.add("text-red-500"),t.setAttribute("aria-pressed","true")}catch(d){Toast.error(d.message||"Wishlist action failed.")}}})}function a(){document.querySelectorAll('input[name="variant"]').forEach(r=>{r.addEventListener("change",()=>{e=r.value;let d=r.dataset.price,l=parseInt(r.dataset.stock||"0");if(d){let Q=document.getElementById("product-price");Q&&window.Templates?.formatPrice&&(Q.textContent=window.Templates.formatPrice(parseFloat(d)))}let p=document.getElementById("stock-status"),f=document.getElementById("add-to-cart-btn"),I=document.getElementById("mobile-stock"),F=document.getElementById("mobile-add-to-cart");p&&(l>10?p.innerHTML='<span class="text-green-600">In Stock</span>':l>0?p.innerHTML=`<span class="text-orange-500">Only ${l} left</span>`:p.innerHTML='<span class="text-red-600">Out of Stock</span>'),f&&(f.disabled=l<=0),F&&(F.disabled=l<=0),I&&(I.textContent=l>0?l>10?"In stock":`${l} available`:"Out of stock")})})}function c(){document.getElementById("main-image")?.addEventListener("click",()=>{})}function u(){let r=window.location.pathname.match(/\/products\/([^\/]+)/);return r?r[1]:null}async function m(t){let r=document.getElementById("product-detail");if(r){Loader.show(r,"skeleton");try{if(n=(await ProductsApi.getProduct(t)).data,!n){window.location.href="/404/";return}document.title=`${n.name} | Bunoraa`,b(n),Ee(n),me(n),await Promise.all([ye(n),ce(n),xe(n),Ce(n)]),_e(),$e(n)}catch(d){console.error("Failed to load product:",d),r.innerHTML='<p class="text-red-500 text-center py-8">Failed to load product. Please try again.</p>'}}}document.addEventListener("currency:changed",async t=>{try{!h&&n&&n.slug?await m(n.slug):location.reload()}catch{}});function b(t){let r=document.getElementById("product-detail");if(!r)return;let d=t.images||[],l=t.image||d[0]?.image||"",p=t.variants&&t.variants.length>0,f=t.stock_quantity>0||t.in_stock,I=t.sale_price&&t.sale_price<t.price;r.innerHTML=`
            <div class="grid grid-cols-1 lg:grid-cols-2 gap-8 lg:gap-12">
                <!-- Gallery -->
                <div id="product-gallery" class="product-gallery">
                    <div class="main-image-container relative rounded-xl overflow-hidden bg-gray-100" style="aspect-ratio: ${t?.aspect?.css||"1/1"};">
                        <img 
                            src="${l}" 
                            alt="${Templates.escapeHtml(t.name)}"
                            loading="lazy"
                            decoding="async"
                            class="main-image w-full h-full object-cover cursor-zoom-in"
                            id="main-product-image"
                        >
                        ${I?`
                            <span class="absolute top-4 left-4 px-3 py-1 bg-red-500 text-white text-sm font-medium rounded-full">
                                Sale
                            </span>
                        `:""}
                        ${f?"":`
                            <span class="absolute top-4 right-4 px-3 py-1 bg-gray-900 text-white text-sm font-medium rounded-full">
                                Out of Stock
                            </span>
                        `}
                    </div>
                    ${d.length>1?`
                        <div class="thumbnails flex gap-2 mt-4 overflow-x-auto pb-2">
                            ${d.map((F,Q)=>`
                                <button 
                                    class="thumbnail flex-shrink-0 w-20 h-20 rounded-lg overflow-hidden border-2 ${Q===0?"border-primary-500":"border-transparent"} hover:border-primary-500 transition-colors"
                                    data-image="${F.image}"
                                    data-index="${Q}"
                                >
                                    <img src="${F.image}" alt="" loading="lazy" decoding="async" class="w-full h-full object-cover">
                                </button>
                            `).join("")}
                        </div>
                    `:""}
                </div>

                <!-- Product Info -->
                <div class="product-info">
                    <!-- Brand -->
                    ${t.brand?`
                        <a href="/products/?brand=${t.brand.id}" class="text-sm text-primary-600 hover:text-primary-700 font-medium">
                            ${Templates.escapeHtml(t.brand.name)}
                        </a>
                    `:""}

                    <!-- Title -->
                    <h1 class="text-2xl md:text-3xl font-bold text-gray-900 mt-2">
                        ${Templates.escapeHtml(t.name)}
                    </h1>

                    <!-- Rating -->
                    ${t.average_rating?`
                        <div class="flex items-center gap-2 mt-3">
                            <div class="flex items-center">
                                ${Templates.renderStars(t.average_rating)}
                            </div>
                            <span class="text-sm text-gray-600">
                                ${t.average_rating.toFixed(1)} (${t.review_count||0} reviews)
                            </span>
                            <a href="#reviews" class="text-sm text-primary-600 hover:text-primary-700">
                                Read reviews
                            </a>
                        </div>
                    `:""}

                    <!-- Price -->
                    <div class="mt-4">
                        ${Price.render({price:t.current_price??t.price_converted??t.price,salePrice:t.sale_price_converted??t.sale_price,size:"large"})}
                    </div>

                    <!-- Short Description -->
                    ${t.short_description?`
                        <p class="mt-4 text-gray-600">${Templates.escapeHtml(t.short_description)}</p>
                    `:""}

                    <!-- Variants -->
                    ${p?P(t.variants):""}

                    <!-- Quantity & Add to Cart -->
                    <div class="mt-6 space-y-4">
                        <div class="flex items-center gap-4">
                            <label class="text-sm font-medium text-gray-700">Quantity:</label>
                            <div class="flex items-center border border-gray-300 rounded-lg">
                                <button 
                                    class="qty-decrease px-3 py-2 text-gray-600 hover:text-gray-900 hover:bg-gray-100 transition-colors"
                                    aria-label="Decrease quantity"
                                >
                                    <svg class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                        <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M20 12H4"/>
                                    </svg>
                                </button>
                                <input 
                                    type="number" 
                                    id="product-quantity"
                                    value="1" 
                                    min="1" 
                                    max="${t.stock_quantity||99}"
                                    class="w-16 text-center border-0 focus:ring-0"
                                >
                                <button 
                                    class="qty-increase px-3 py-2 text-gray-600 hover:text-gray-900 hover:bg-gray-100 transition-colors"
                                    aria-label="Increase quantity"
                                >
                                    <svg class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                        <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 4v16m8-8H4"/>
                                    </svg>
                                </button>
                            </div>
                            ${t.stock_quantity?`
                                <div id="stock-status" class="text-sm text-gray-500">${t.stock_quantity>10?"In stock":t.stock_quantity+" available"}</div>
                            `:'<div id="stock-status" class="text-red-600">Out of Stock</div>'}
                        </div>

                        <div class="flex gap-3">
                            <button 
                                id="add-to-cart-btn"
                                class="flex-1 px-6 py-3 bg-primary-600 text-white font-semibold rounded-lg hover:bg-primary-700 disabled:bg-gray-400 disabled:cursor-not-allowed transition-colors flex items-center justify-center gap-2"
                                ${f?"":"disabled"}
                            >
                                <svg class="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                    <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M3 3h2l.4 2M7 13h10l4-8H5.4M7 13L5.4 5M7 13l-2.293 2.293c-.63.63-.184 1.707.707 1.707H17m0 0a2 2 0 100 4 2 2 0 000-4zm-8 2a2 2 0 11-4 0 2 2 0 014 0z"/>
                                </svg>
                                ${f?"Add to Cart":"Out of Stock"}
                            </button>
                            <button 
                                id="add-to-wishlist-btn"
                                class="px-4 py-3 border border-gray-300 rounded-lg hover:bg-gray-50 transition-colors"
                                aria-label="Add to wishlist"
                                data-product-id="${t.id}"
                            >
                                <svg class="w-6 h-6 text-gray-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                    <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M4.318 6.318a4.5 4.5 0 000 6.364L12 20.364l7.682-7.682a4.5 4.5 0 00-6.364-6.364L12 7.636l-1.318-1.318a4.5 4.5 0 00-6.364 0z"/>
                                </svg>
                            </button>
                        </div>
                    </div>

                    <!-- Product Meta -->
                    <div class="mt-6 pt-6 border-t border-gray-200 space-y-3 text-sm">
                        ${t.sku?`
                            <div class="flex">
                                <span class="text-gray-500 w-24">SKU:</span>
                                <span class="text-gray-900">${Templates.escapeHtml(t.sku)}</span>
                            </div>
                        `:""}
                        ${t.category?`
                            <div class="flex">
                                <span class="text-gray-500 w-24">Category:</span>
                                <a href="/categories/${t.category.slug}/" class="text-primary-600 hover:text-primary-700">
                                    ${Templates.escapeHtml(t.category.name)}
                                </a>
                            </div>
                        `:""}
                        ${t.tags&&t.tags.length?`
                            <div class="flex">
                                <span class="text-gray-500 w-24">Tags:</span>
                                <div class="flex flex-wrap gap-1">
                                    ${t.tags.map(F=>`
                                        <a href="/products/?tag=${F.slug}" class="px-2 py-0.5 bg-gray-100 text-gray-600 rounded hover:bg-gray-200 transition-colors">
                                            ${Templates.escapeHtml(F.name)}
                                        </a>
                                    `).join("")}
                                </div>
                            </div>
                        `:""}
                    </div>

                    <!-- Share -->
                    <div class="mt-6 pt-6 border-t border-gray-200">
                        <span class="text-sm text-gray-500">Share:</span>
                        <div class="flex gap-2 mt-2">
                            <button class="share-btn p-2 rounded-full bg-gray-100 hover:bg-gray-200 transition-colors" data-platform="facebook" aria-label="Share on Facebook">
                                <svg class="w-5 h-5 text-[#1877F2]" fill="currentColor" viewBox="0 0 24 24"><path d="M24 12.073c0-6.627-5.373-12-12-12s-12 5.373-12 12c0 5.99 4.388 10.954 10.125 11.854v-8.385H7.078v-3.47h3.047V9.43c0-3.007 1.792-4.669 4.533-4.669 1.312 0 2.686.235 2.686.235v2.953H15.83c-1.491 0-1.956.925-1.956 1.874v2.25h3.328l-.532 3.47h-2.796v8.385C19.612 23.027 24 18.062 24 12.073z"/></svg>
                            </button>
                            <button class="share-btn p-2 rounded-full bg-gray-100 hover:bg-gray-200 transition-colors" data-platform="twitter" aria-label="Share on Twitter">
                                <svg class="w-5 h-5 text-[#1DA1F2]" fill="currentColor" viewBox="0 0 24 24"><path d="M23.953 4.57a10 10 0 01-2.825.775 4.958 4.958 0 002.163-2.723c-.951.555-2.005.959-3.127 1.184a4.92 4.92 0 00-8.384 4.482C7.69 8.095 4.067 6.13 1.64 3.162a4.822 4.822 0 00-.666 2.475c0 1.71.87 3.213 2.188 4.096a4.904 4.904 0 01-2.228-.616v.06a4.923 4.923 0 003.946 4.827 4.996 4.996 0 01-2.212.085 4.936 4.936 0 004.604 3.417 9.867 9.867 0 01-6.102 2.105c-.39 0-.779-.023-1.17-.067a13.995 13.995 0 007.557 2.209c9.053 0 13.998-7.496 13.998-13.985 0-.21 0-.42-.015-.63A9.935 9.935 0 0024 4.59z"/></svg>
                            </button>
                            <button class="share-btn p-2 rounded-full bg-gray-100 hover:bg-gray-200 transition-colors" data-platform="pinterest" aria-label="Share on Pinterest">
                                <svg class="w-5 h-5 text-[#E60023]" fill="currentColor" viewBox="0 0 24 24"><path d="M12 0C5.373 0 0 5.372 0 12c0 5.084 3.163 9.426 7.627 11.174-.105-.949-.2-2.405.042-3.441.218-.937 1.407-5.965 1.407-5.965s-.359-.719-.359-1.782c0-1.668.967-2.914 2.171-2.914 1.023 0 1.518.769 1.518 1.69 0 1.029-.655 2.568-.994 3.995-.283 1.194.599 2.169 1.777 2.169 2.133 0 3.772-2.249 3.772-5.495 0-2.873-2.064-4.882-5.012-4.882-3.414 0-5.418 2.561-5.418 5.207 0 1.031.397 2.138.893 2.738a.36.36 0 01.083.345l-.333 1.36c-.053.22-.174.267-.402.161-1.499-.698-2.436-2.889-2.436-4.649 0-3.785 2.75-7.262 7.929-7.262 4.163 0 7.398 2.967 7.398 6.931 0 4.136-2.607 7.464-6.227 7.464-1.216 0-2.359-.631-2.75-1.378l-.748 2.853c-.271 1.043-1.002 2.35-1.492 3.146C9.57 23.812 10.763 24 12 24c6.627 0 12-5.373 12-12 0-6.628-5.373-12-12-12z"/></svg>
                            </button>
                            <button class="share-btn p-2 rounded-full bg-gray-100 hover:bg-gray-200 transition-colors" data-platform="copy" aria-label="Copy link">
                                <svg class="w-5 h-5 text-gray-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                    <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M8 5H6a2 2 0 00-2 2v12a2 2 0 002 2h10a2 2 0 002-2v-1M8 5a2 2 0 002 2h2a2 2 0 002-2M8 5a2 2 0 012-2h2a2 2 0 012 2m0 0h2a2 2 0 012 2v3m2 4H10m0 0l3-3m-3 3l3 3"/>
                                </svg>
                            </button>
                        </div>
                    </div>
                </div>
            </div>

            <!-- Product Tabs -->
            <div class="mt-12" data-tabs data-variant="underline" id="product-tabs">
                <div class="border-b border-gray-200">
                    <nav class="flex -mb-px">
                        <button data-tab class="px-6 py-4 text-sm font-medium text-gray-600 hover:text-gray-900 border-b-2 border-transparent">
                            Description
                        </button>
                        ${t.specifications&&Object.keys(t.specifications).length?`
                            <button data-tab class="px-6 py-4 text-sm font-medium text-gray-600 hover:text-gray-900 border-b-2 border-transparent">
                                Specifications
                            </button>
                        `:""}
                        <button data-tab class="px-6 py-4 text-sm font-medium text-gray-600 hover:text-gray-900 border-b-2 border-transparent">
                            Reviews (${t.review_count||0})
                        </button>
                    </nav>
                </div>
                <div class="py-6">
                    <div data-tab-panel>
                        <div class="prose max-w-none">
                            ${t.description||'<p class="text-gray-500">No description available.</p>'}
                        </div>
                    </div>
                    ${t.specifications&&Object.keys(t.specifications).length?`
                        <div data-tab-panel>
                            <table class="w-full">
                                <tbody>
                                    ${Object.entries(t.specifications).map(([F,Q])=>`
                                        <tr class="border-b border-gray-100">
                                            <td class="py-3 text-sm font-medium text-gray-500 w-1/3">${Templates.escapeHtml(F)}</td>
                                            <td class="py-3 text-sm text-gray-900">${Templates.escapeHtml(String(Q))}</td>
                                        </tr>
                                    `).join("")}
                                </tbody>
                            </table>
                        </div>
                    `:""}
                    <div data-tab-panel id="reviews">
                        <div id="reviews-container">
                            <!-- Reviews loaded here -->
                        </div>
                    </div>
                </div>
            </div>
        `,D(),J(),le(),be(),fe(),ke(),Tabs.init(),v(t)}function v(t){let r=document.getElementById("mobile-sticky-atc-js");if(r){let d=r.querySelector(".font-semibold");d&&(d.innerHTML=t.sale_price?Templates.formatPrice(t.sale_price)+' <span class="text-sm line-through text-gray-400">'+Templates.formatPrice(t.price)+"</span>":Templates.formatPrice(t.price));let l=document.getElementById("mobile-stock-js");l&&(l.textContent=t.stock_quantity>0?t.stock_quantity>10?"In stock":t.stock_quantity+" available":"Out of stock");let p=document.getElementById("mobile-add-to-cart-js");p&&(p.disabled=t.stock_quantity<=0)}else{r=document.createElement("div"),r.id="mobile-sticky-atc-js",r.className="fixed inset-x-4 bottom-4 z-40 lg:hidden",r.innerHTML=`
                <div class="bg-white shadow-lg rounded-xl p-3 flex items-center gap-3">
                    <div class="flex-1">
                        <div class="text-sm text-gray-500">${t.sale_price?"Now":""}</div>
                        <div class="font-semibold text-lg ${t.sale_price?"text-red-600":""}">${t.sale_price?Templates.formatPrice(t.sale_price)+' <span class="text-sm line-through text-gray-400">'+Templates.formatPrice(t.price)+"</span>":Templates.formatPrice(t.price)}</div>
                        <div id="mobile-stock-js" class="text-xs text-gray-500">${t.stock_quantity>0?t.stock_quantity>10?"In stock":t.stock_quantity+" available":"Out of stock"}</div>
                    </div>
                    ${t.stock_quantity>0?'<button id="mobile-add-to-cart-js" class="bg-primary-600 text-white px-4 py-2 rounded-lg font-semibold">Add</button>':'<button class="bg-gray-300 text-gray-500 px-4 py-2 rounded-lg font-semibold cursor-not-allowed" disabled>Out</button>'}
                </div>
            `,document.body.appendChild(r);let d=document.getElementById("mobile-add-to-cart-js");d&&d.addEventListener("click",()=>document.getElementById("add-to-cart-btn")?.click())}}function P(t){let r={};return t.forEach(d=>{r[d.attribute_name]||(r[d.attribute_name]=[]),r[d.attribute_name].push(d)}),Object.entries(r).map(([d,l])=>`
            <div class="mt-6">
                <label class="text-sm font-medium text-gray-700">${Templates.escapeHtml(d)}:</label>
                <div class="flex flex-wrap gap-2 mt-2" role="radiogroup" aria-label="${Templates.escapeHtml(d)}">
                    ${l.map((p,f)=>`
                        <button 
                            class="variant-btn px-4 py-2 border rounded-lg text-sm transition-colors ${f===0?"border-primary-500 bg-primary-50 text-primary-700":"border-gray-300 hover:border-gray-400"}"
                            role="radio"
                            aria-checked="${f===0?"true":"false"}"
                            data-variant-id="${p.id}"
                            data-price="${p.price_converted??p.price??""}"
                            data-stock="${p.stock_quantity||0}"
                            tabindex="0"
                        >
                            ${Templates.escapeHtml(p.value)}
                            ${(p.price_converted??p.price)&&p.price!==n.price?`
                                <span class="text-xs text-gray-500">(${Templates.formatPrice(p.price_converted??p.price)})</span>
                            `:""}
                        </button>
                    `).join("")}
                </div>
            </div>
        `).join("")}function D(){let t=document.querySelectorAll(".thumbnail"),r=document.getElementById("main-product-image"),d=0;t.forEach((l,p)=>{l.setAttribute("tabindex","0"),l.addEventListener("click",()=>{t.forEach(f=>f.classList.remove("border-primary-500")),l.classList.add("border-primary-500"),r.src=l.dataset.image||l.dataset.src,d=p}),l.addEventListener("keydown",f=>{if(f.key==="Enter"||f.key===" ")f.preventDefault(),l.click();else if(f.key==="ArrowRight"){f.preventDefault();let I=t[(p+1)%t.length];I.focus(),I.click()}else if(f.key==="ArrowLeft"){f.preventDefault();let I=t[(p-1+t.length)%t.length];I.focus(),I.click()}})}),r?.addEventListener("click",()=>{let l=n.images?.map(F=>F.image)||[n.image],p=parseInt(document.querySelector(".thumbnail.border-primary-500")?.dataset.index)||0,I=(n.images||[]).map(F=>({type:F.type||(F.video_url?"video":"image"),src:F.video_url||F.model_url||F.image})).map(F=>{if(F.type==="video")return`<div class="w-full h-full max-h-[70vh]"><video controls class="w-full h-full object-contain"><source src="${F.src}" type="video/mp4">Your browser does not support video.</video></div>`;if(F.type==="model"){if(!window.customElements||!window["model-viewer"]){let Q=document.createElement("script");Q.type="module",Q.src="https://unpkg.com/@google/model-viewer/dist/model-viewer.min.js",document.head.appendChild(Q)}return`<div class="w-full h-full max-h-[70vh]"><model-viewer src="${F.src}" camera-controls ar ar-modes="webxr scene-viewer quick-look" class="w-full h-full"></model-viewer></div>`}return`<div class="w-full h-full max-h-[70vh] flex items-center justify-center"><img src="${F.src}" class="max-w-full max-h-[70vh] object-contain" alt="${Templates.escapeHtml(n.name)}"></div>`}).join("");Modal.open({title:Templates.escapeHtml(n.name),content:`<div class="space-y-2">${I}</div>`,size:"xl"})})}function J(){let t=document.getElementById("product-quantity"),r=document.querySelector(".qty-decrease"),d=document.querySelector(".qty-increase");r?.addEventListener("click",()=>{let p=parseInt(t.value)||1;p>1&&(t.value=p-1)}),d?.addEventListener("click",()=>{let p=parseInt(t.value)||1,f=parseInt(t.max)||99;p<f&&(t.value=p+1)});let l=document.querySelectorAll(".variant-btn");if(l.forEach(p=>{p.addEventListener("click",()=>{if(l.forEach(ae=>{ae.classList.remove("border-primary-500","bg-primary-50","text-primary-700"),ae.classList.add("border-gray-300"),ae.setAttribute("aria-checked","false")}),p.classList.add("border-primary-500","bg-primary-50","text-primary-700"),p.classList.remove("border-gray-300"),p.setAttribute("aria-checked","true"),e=p.dataset.variantId,p.dataset.price){let ae=document.querySelector(".product-info .mt-4");ae&&(ae.innerHTML=Price.render({price:parseFloat(p.dataset.price),size:"large"}))}let f=parseInt(p.dataset.stock||"0"),I=document.getElementById("stock-status"),F=document.getElementById("add-to-cart-btn"),Q=document.getElementById("mobile-stock"),K=document.getElementById("mobile-add-to-cart");I&&(f>10?I.innerHTML='<span class="text-green-600 flex items-center"><svg class="w-5 h-5 mr-1" fill="currentColor" viewBox="0 0 20 20"><path fill-rule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zm3.707-9.293a1 1 0 00-1.414-1.414L9 10.586 7.707 9.293a1 1 0 00-1.414 1.414l2 2a1 1 0 001.414 0l4-4z" clip-rule="evenodd"></path></svg>In Stock</span>':f>0?I.innerHTML=`<span class="text-orange-500 flex items-center"><svg class="w-5 h-5 mr-1" fill="currentColor" viewBox="0 0 20 20"><path fill-rule="evenodd" d="M8.257 3.099c.765-1.36 2.722-1.36 3.486 0l5.58 9.92c.75 1.334-.213 2.98-1.742 2.98H4.42c-1.53 0-2.493-1.646-1.743-2.98l5.58-9.92zM11 13a1 1 0 11-2 0 1 1 0 012 0zm-1-8a1 1 0 00-1 1v3a1 1 0 002 0V6a1 1 0 00-1-1z" clip-rule="evenodd"></path></svg>Only ${f} left</span>`:I.innerHTML='<span class="text-red-600 flex items-center"><svg class="w-5 h-5 mr-1" fill="currentColor" viewBox="0 0 20 20"><path fill-rule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zM8.707 7.293a1 1 0 00-1.414 1.414L8.586 10l-1.293 1.293a1 1 0 101.414 1.414L10 11.414l1.293 1.293a1 1 0 001.414-1.414L11.414 10l1.293-1.293a1 1 0 00-1.414-1.414L10 8.586 8.707 7.293z" clip-rule="evenodd"></path></svg>Out of Stock</span>'),t&&(t.max=Math.max(f,1),parseInt(t.value)>parseInt(t.max)&&(t.value=t.max)),Q&&(Q.textContent=f>0?f>10?"In stock":`${f} available`:"Out of stock"),F&&(F.disabled=f<=0),K&&(K.disabled=f<=0)})}),l.length>0){let p=l[0];p.setAttribute("aria-checked","true"),e=p.dataset.variantId}}function le(){let t=document.getElementById("add-to-cart-btn"),r=document.getElementById("mobile-add-to-cart");if(!t&&!r)return;let d=async l=>{let p=parseInt(document.getElementById("product-quantity")?.value)||1,f=document.getElementById("stock-status");if(!!document.querySelector(".variant-btn")&&!e){Toast.warning("Please select a variant before adding to cart.");return}l.disabled=!0;let F=l.innerHTML;l.innerHTML='<svg class="animate-spin h-5 w-5" fill="none" viewBox="0 0 24 24"><circle class="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" stroke-width="4"></circle><path class="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path></svg>';try{let Q=document.querySelector(`.variant-btn[data-variant-id="${e}"]`);if((Q?parseInt(Q.dataset.stock||"0"):n.stock_quantity||0)<=0){Toast.error("This variant is out of stock.");return}await CartApi.addItem(n.id,p,e||null),Toast.success("Added to cart!"),document.dispatchEvent(new CustomEvent("cart:updated"))}catch(Q){Toast.error(Q.message||"Failed to add to cart.")}finally{l.disabled=!1,l.innerHTML=F}};t?.addEventListener("click",()=>d(t)),r?.addEventListener("click",()=>d(r))}function be(){let t=document.getElementById("add-to-wishlist-btn");t&&t.addEventListener("click",async()=>{if(!AuthApi.isAuthenticated()){Toast.warning("Please login to add items to your wishlist."),window.location.href="/account/login/?next="+encodeURIComponent(window.location.pathname);return}try{await WishlistApi.addItem(n.id),Toast.success("Added to wishlist!"),t.querySelector("svg").setAttribute("fill","currentColor"),t.classList.add("text-red-500"),t.setAttribute("aria-pressed","true")}catch(r){r.message?.includes("already")?Toast.info("This item is already in your wishlist."):Toast.error(r.message||"Failed to add to wishlist.")}})}function ke(){let t=document.querySelectorAll(".share-btn"),r=encodeURIComponent(window.location.href),d=encodeURIComponent(n?.name||document.title);t.forEach(l=>{l.addEventListener("click",()=>{let p=l.dataset.platform,f="";switch(p){case"facebook":f=`https://www.facebook.com/sharer/sharer.php?u=${r}`;break;case"twitter":f=`https://twitter.com/intent/tweet?url=${r}&text=${d}`;break;case"pinterest":let I=encodeURIComponent(n?.image||"");f=`https://pinterest.com/pin/create/button/?url=${r}&media=${I}&description=${d}`;break;case"copy":navigator.clipboard.writeText(window.location.href).then(()=>Toast.success("Link copied to clipboard!")).catch(()=>Toast.error("Failed to copy link."));return}f&&window.open(f,"_blank","width=600,height=400")})})}function fe(){let t=document.getElementById("add-to-compare-btn");t&&t.addEventListener("click",async()=>{if(typeof AuthApi<"u"&&!AuthApi.isAuthenticated()){Toast.warning("Please login to compare products."),window.location.href="/account/login/?next="+encodeURIComponent(window.location.pathname);return}let r=n?.id||document.getElementById("product-detail")?.dataset.productId;if(r)try{let d=await ApiClient.post("/compare/",{product_id:r},{requiresAuth:!0});d.success?(Toast.success(d.message||"Added to compare"),t.setAttribute("aria-pressed","true"),t.classList.add("text-primary-600"),t.querySelector("svg")?.setAttribute("fill","currentColor")):Toast.error(d.message||"Failed to add to compare")}catch(d){try{let l="b_compare",p=JSON.parse(localStorage.getItem(l)||"[]");if(!p.includes(r)){p.push(r),localStorage.setItem(l,JSON.stringify(p)),Toast.success("Added to compare (local)"),t.setAttribute("aria-pressed","true"),t.classList.add("text-primary-600");return}Toast.info("Already in compare list")}catch{Toast.error(d.message||"Failed to add to compare")}}})}function we(){let t=document.getElementById("add-to-compare-btn");t&&t.addEventListener("click",async r=>{r.preventDefault();let d=document.getElementById("product-detail")?.dataset.productId;if(d)try{if(typeof AuthApi<"u"&&!AuthApi.isAuthenticated()){Toast.warning("Please login to compare products."),window.location.href="/account/login/?next="+encodeURIComponent(window.location.pathname);return}let l=await ApiClient.post("/compare/",{product_id:d},{requiresAuth:!0});l.success?(Toast.success(l.message||"Added to compare"),t.setAttribute("aria-pressed","true"),t.classList.add("text-primary-600"),t.querySelector("svg")?.setAttribute("fill","currentColor")):Toast.error(l.message||"Failed to add to compare")}catch(l){try{let p="b_compare",f=JSON.parse(localStorage.getItem(p)||"[]");if(!f.includes(d)){f.push(d),localStorage.setItem(p,JSON.stringify(f)),Toast.success("Added to compare (local)"),t.setAttribute("aria-pressed","true"),t.classList.add("text-primary-600");return}Toast.info("Already in compare list")}catch{Toast.error(l.message||"Failed to add to compare")}}})}async function ye(t){let r=document.getElementById("breadcrumbs");if(!r)return;let d=[{label:"Home",url:"/"},{label:"Products",url:"/products/"}];if(t.category)try{((await CategoriesAPI.getBreadcrumbs(t.category.id)).data||[]).forEach(f=>{d.push({label:f.name,url:`/categories/${f.slug}/`})})}catch{d.push({label:t.category.name,url:`/categories/${t.category.slug}/`})}d.push({label:t.name}),r.innerHTML=Breadcrumb.render(d)}async function ce(t){let r=document.getElementById("related-products");if(r)try{let l=(await ProductsAPI.getRelated(t.id,{limit:4})).data||[];if(l.length===0){r.innerHTML="";return}r.innerHTML=`
                <h2 class="text-2xl font-bold text-gray-900 mb-6">You may also like</h2>
                <div class="grid grid-cols-2 md:grid-cols-4 gap-4 md:gap-6">
                    ${l.map(p=>ProductCard.render(p)).join("")}
                </div>
            `,ProductCard.bindEvents(r)}catch(d){console.error("Failed to load related products:",d),r.innerHTML=""}}async function xe(t){let r=document.getElementById("reviews-container");if(r){Loader.show(r,"spinner");try{let l=(await ProductsAPI.getReviews(t.id)).data||[];r.innerHTML=`
                <!-- Review Summary -->
                <div class="flex flex-col md:flex-row gap-8 mb-8 pb-8 border-b border-gray-200">
                    <div class="text-center">
                        <div class="text-5xl font-bold text-gray-900">${t.average_rating?.toFixed(1)||"0.0"}</div>
                        <div class="flex justify-center mt-2">
                            ${Templates.renderStars(t.average_rating||0)}
                        </div>
                        <div class="text-sm text-gray-500 mt-1">${t.review_count||0} reviews</div>
                    </div>
                    ${AuthAPI.isAuthenticated()?`
                        <div class="flex-1">
                            <button id="write-review-btn" class="px-6 py-3 bg-primary-600 text-white font-semibold rounded-lg hover:bg-primary-700 transition-colors">
                                Write a Review
                            </button>
                        </div>
                    `:`
                        <div class="flex-1">
                            <p class="text-gray-600">
                                <a href="/account/login/?next=${encodeURIComponent(window.location.pathname)}" class="text-primary-600 hover:text-primary-700">Sign in</a> 
                                to write a review.
                            </p>
                        </div>
                    `}
                </div>

                <!-- Reviews List -->
                ${l.length>0?`
                    <div class="space-y-6">
                        ${l.map(p=>`
                            <div class="border-b border-gray-100 pb-6">
                                <div class="flex items-start gap-4">
                                    <div class="flex-shrink-0 w-10 h-10 bg-gray-200 rounded-full flex items-center justify-center">
                                        <span class="text-gray-600 font-medium">${(p.user?.first_name?.[0]||p.user?.email?.[0]||"U").toUpperCase()}</span>
                                    </div>
                                    <div class="flex-1">
                                        <div class="flex items-center gap-2">
                                            <span class="font-medium text-gray-900">${Templates.escapeHtml(p.user?.first_name||"Anonymous")}</span>
                                            ${p.verified_purchase?`
                                                <span class="px-2 py-0.5 bg-green-100 text-green-700 text-xs rounded">Verified Purchase</span>
                                            `:""}
                                        </div>
                                        <div class="flex items-center gap-2 mt-1">
                                            ${Templates.renderStars(p.rating)}
                                            <span class="text-sm text-gray-500">${Templates.formatDate(p.created_at)}</span>
                                        </div>
                                        ${p.title?`<h4 class="font-medium text-gray-900 mt-2">${Templates.escapeHtml(p.title)}</h4>`:""}
                                        <p class="text-gray-600 mt-2">${Templates.escapeHtml(p.comment)}</p>
                                    </div>
                                </div>
                            </div>
                        `).join("")}
                    </div>
                `:`
                    <p class="text-gray-500 text-center py-8">No reviews yet. Be the first to review this product!</p>
                `}
            `,document.getElementById("write-review-btn")?.addEventListener("click",()=>{pe(t)})}catch(d){console.error("Failed to load reviews:",d),r.innerHTML='<p class="text-red-500">Failed to load reviews.</p>'}}}function pe(t){Modal.open({title:"Write a Review",content:`
                <form id="review-form" class="space-y-4">
                    <div>
                        <label class="block text-sm font-medium text-gray-700 mb-2">Rating</label>
                        <div class="flex gap-1" id="rating-stars">
                            ${[1,2,3,4,5].map(l=>`
                                <button type="button" class="rating-star text-gray-300 hover:text-yellow-400" data-rating="${l}">
                                    <svg class="w-8 h-8" fill="currentColor" viewBox="0 0 20 20">
                                        <path d="M9.049 2.927c.3-.921 1.603-.921 1.902 0l1.07 3.292a1 1 0 00.95.69h3.462c.969 0 1.371 1.24.588 1.81l-2.8 2.034a1 1 0 00-.364 1.118l1.07 3.292c.3.921-.755 1.688-1.54 1.118l-2.8-2.034a1 1 0 00-1.175 0l-2.8 2.034c-.784.57-1.838-.197-1.539-1.118l1.07-3.292a1 1 0 00-.364-1.118L2.98 8.72c-.783-.57-.38-1.81.588-1.81h3.461a1 1 0 00.951-.69l1.07-3.292z"/>
                                    </svg>
                                </button>
                            `).join("")}
                        </div>
                        <input type="hidden" id="review-rating" value="0">
                    </div>
                    <div>
                        <label for="review-title" class="block text-sm font-medium text-gray-700 mb-1">Title (optional)</label>
                        <input type="text" id="review-title" class="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-primary-500 focus:border-primary-500">
                    </div>
                    <div>
                        <label for="review-comment" class="block text-sm font-medium text-gray-700 mb-1">Your Review</label>
                        <textarea id="review-comment" rows="4" class="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-primary-500 focus:border-primary-500" required></textarea>
                    </div>
                </form>
            `,confirmText:"Submit Review",onConfirm:async()=>{let l=parseInt(document.getElementById("review-rating").value),p=document.getElementById("review-title").value.trim(),f=document.getElementById("review-comment").value.trim();if(!l||l<1)return Toast.error("Please select a rating."),!1;if(!f)return Toast.error("Please write a review."),!1;try{return await ProductsAPI.createReview(t.id,{rating:l,title:p,comment:f}),Toast.success("Thank you for your review!"),xe(t),!0}catch(I){return Toast.error(I.message||"Failed to submit review."),!1}}});let r=document.querySelectorAll(".rating-star"),d=document.getElementById("review-rating");r.forEach(l=>{l.addEventListener("click",()=>{let p=parseInt(l.dataset.rating);d.value=p,r.forEach((f,I)=>{I<p?(f.classList.remove("text-gray-300"),f.classList.add("text-yellow-400")):(f.classList.add("text-gray-300"),f.classList.remove("text-yellow-400"))})})})}function Ee(t){try{document.title=`${t.name} | Bunoraa`;let r=t.meta_description||t.short_description||"";document.querySelector('meta[name="description"]')?.setAttribute("content",r),document.querySelector('meta[property="og:title"]')?.setAttribute("content",t.meta_title||t.name),document.querySelector('meta[property="og:description"]')?.setAttribute("content",r);let d=t.images?.[0]?.image||t.image;d&&document.querySelector('meta[property="og:image"]')?.setAttribute("content",d),document.querySelector('meta[name="twitter:title"]')?.setAttribute("content",t.meta_title||t.name),document.querySelector('meta[name="twitter:description"]')?.setAttribute("content",r)}catch{}}function me(t){try{let r=document.querySelector('script[type="application/ld+json"][data-ld="product"]');if(!r)return;let d={"@context":"https://schema.org","@type":"Product",name:t.name,image:(t.images||[]).map(l=>l.image||l),description:t.short_description||t.description||"",sku:t.sku||"",offers:{"@type":"Offer",url:window.location.href,priceCurrency:t.currency||window.BUNORAA_PRODUCT?.currency||"BDT",price:t.current_price||t.price}};r.textContent=JSON.stringify(d)}catch{}}async function Ce(t){let r=document.getElementById("related-products");if(r)try{let[d,l,p]=await Promise.all([ProductsApi.getRecommendations(t.id,"frequently_bought_together",3),ProductsApi.getRecommendations(t.id,"similar",4),ProductsApi.getRecommendations(t.id,"you_may_also_like",6)]);if(d.success&&d.data?.length){let f=`
                    <section class="mt-8">
                        <h3 class="text-lg font-semibold mb-4">Frequently Bought Together</h3>
                        <div class="grid grid-cols-3 gap-3">${(d.data||[]).map(I=>ProductCard.render(I)).join("")}</div>
                    </section>
                `;r.insertAdjacentHTML("beforeend",f)}if(l.success&&l.data?.length){let f=`
                    <section class="mt-8">
                        <h3 class="text-lg font-semibold mb-4">Similar Products</h3>
                        <div class="grid grid-cols-2 md:grid-cols-4 gap-4">${(l.data||[]).map(I=>ProductCard.render(I)).join("")}</div>
                    </section>
                `;r.insertAdjacentHTML("beforeend",f)}if(p.success&&p.data?.length){let f=`
                    <section class="mt-8">
                        <h3 class="text-lg font-semibold mb-4">You May Also Like</h3>
                        <div class="grid grid-cols-2 md:grid-cols-4 gap-4">${(p.data||[]).map(I=>ProductCard.render(I)).join("")}</div>
                    </section>
                `;r.insertAdjacentHTML("beforeend",f)}ProductCard.bindEvents(r)}catch{}}function _e(){if(!document.getElementById("product-detail")||typeof IntersectionObserver>"u")return;let r=new IntersectionObserver(d=>{d.forEach(l=>{if(!l.isIntersecting)return;let p=l.target;p.id,p.id==="reviews"||p.id,r.unobserve(p)})},{rootMargin:"200px"});document.querySelectorAll("#related-products, #reviews").forEach(d=>{r.observe(d)})}async function $e(t){try{await ProductsAPI.trackView(t.id)}catch{}}function x(){n=null,e=null,s=null,o=!1}return{init:L,destroy:x}})();window.ProductPage=fr;Yr=fr});var vr={};te(vr,{default:()=>Gr});var xr,Gr,wr=ee(()=>{xr=(function(){"use strict";let n="",e=1,s={},o=null,h=!1;async function w(){if(h)return;h=!0;let y=document.getElementById("search-results")||document.getElementById("products-grid");if(y&&y.querySelector(".product-card, [data-product-id]")){z(),C(),k(),_();return}n=Y(),s=Z(),e=parseInt(new URLSearchParams(window.location.search).get("page"))||1,await L(),z(),C(),k(),_()}function _(){let y=document.getElementById("view-grid"),i=document.getElementById("view-list");y?.addEventListener("click",()=>{y.classList.add("bg-primary-100","text-primary-700"),y.classList.remove("text-gray-400"),i?.classList.remove("bg-primary-100","text-primary-700"),i?.classList.add("text-gray-400"),Storage?.set("productViewMode","grid"),L()}),i?.addEventListener("click",()=>{i.classList.add("bg-primary-100","text-primary-700"),i.classList.remove("text-gray-400"),y?.classList.remove("bg-primary-100","text-primary-700"),y?.classList.add("text-gray-400"),Storage?.set("productViewMode","list"),L()})}async function L(){let y=document.getElementById("search-results")||document.getElementById("products-grid"),i=document.getElementById("results-count")||document.getElementById("product-count");if(y){o&&o.abort(),o=new AbortController,Loader.show(y,"skeleton");try{let a={page:e,pageSize:12,...s};if(n&&(a.search=n),window.location.pathname==="/categories/"){await A();return}let u=await ProductsApi.getProducts(a),m=Array.isArray(u)?u:u.data||u.results||[],b=u.meta||{};i&&(n?i.textContent=`${b.total||m.length} results for "${Templates.escapeHtml(n)}"`:i.textContent=`${b.total||m.length} products`),H(m,b),await U()}catch(a){if(a.name==="AbortError")return;console.error("Failed to load products:",a),y.innerHTML='<p class="text-red-500 text-center py-8">Failed to load products. Please try again.</p>'}}}async function A(){let y=document.getElementById("search-results")||document.getElementById("products-grid"),i=document.getElementById("results-count")||document.getElementById("product-count"),a=document.getElementById("page-title");if(y)try{let c=await CategoriesApi.getCategories({limit:50}),u=Array.isArray(c)?c:c.data||c.results||[];if(a&&(a.textContent="All Categories"),i&&(i.textContent=`${u.length} categories`),u.length===0){y.innerHTML=`
                    <div class="text-center py-16">
                        <svg class="w-24 h-24 text-gray-300 mx-auto mb-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="1.5" d="M19 11H5m14 0a2 2 0 012 2v6a2 2 0 01-2 2H5a2 2 0 01-2-2v-6a2 2 0 012-2m14 0V9a2 2 0 00-2-2M5 11V9a2 2 0 012-2m0 0V5a2 2 0 012-2h6a2 2 0 012 2v2M7 7h10"/>
                        </svg>
                        <h2 class="text-2xl font-bold text-gray-900 mb-2">No categories found</h2>
                        <p class="text-gray-600">Check back later for new categories.</p>
                    </div>
                `;return}y.innerHTML=`
                <div class="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-4 xl:grid-cols-5 gap-6">
                    ${u.map(m=>`
                        <a href="/categories/${m.slug}/" class="group bg-white rounded-xl shadow-sm hover:shadow-lg transition-all duration-300 overflow-hidden">
                            <div class="relative overflow-hidden" style="aspect-ratio: ${product?.aspect?.css||"1/1"};">
                                ${m.image?`
                                    <img src="${m.image}" alt="${Templates.escapeHtml(m.name)}" class="w-full h-full object-cover group-hover:scale-105 transition-transform duration-300">
                                `:`
                                    <div class="w-full h-full flex items-center justify-center">
                                        <svg class="w-16 h-16 text-gray-300" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="1.5" d="M19 11H5m14 0a2 2 0 012 2v6a2 2 0 01-2 2H5a2 2 0 01-2-2v-6a2 2 0 012-2m14 0V9a2 2 0 00-2-2M5 11V9a2 2 0 012-2m0 0V5a2 2 0 012-2h6a2 2 0 012 2v2M7 7h10"/>
                                        </svg>
                                    </div>
                                `}
                            </div>
                            <div class="p-4 text-center">
                                <h3 class="font-semibold text-gray-900 group-hover:text-primary-600 transition-colors">${Templates.escapeHtml(m.name)}</h3>
                                ${m.product_count?`<p class="text-sm text-gray-500 mt-1">${m.product_count} products</p>`:""}
                            </div>
                        </a>
                    `).join("")}
                </div>
            `}catch(c){console.error("Failed to load categories:",c),y.innerHTML='<p class="text-red-500 text-center py-8">Failed to load categories. Please try again.</p>'}}function Y(){return new URLSearchParams(window.location.search).get("q")||""}function Z(){let y=new URLSearchParams(window.location.search),i={};return y.get("category")&&(i.category=y.get("category")),y.get("min_price")&&(i.minPrice=y.get("min_price")),y.get("max_price")&&(i.maxPrice=y.get("max_price")),y.get("ordering")&&(i.ordering=y.get("ordering")),y.get("in_stock")&&(i.inStock=y.get("in_stock")==="true"),y.get("sale")&&(i.onSale=y.get("sale")==="true"),y.get("featured")&&(i.featured=y.get("featured")==="true"),i}function z(){let y=document.getElementById("search-form"),i=document.getElementById("search-input");i&&(i.value=n),y?.addEventListener("submit",u=>{u.preventDefault();let m=i?.value.trim();m&&(n=m,e=1,S(),E())});let a=document.getElementById("search-suggestions"),c=null;i?.addEventListener("input",u=>{let m=u.target.value.trim();if(clearTimeout(c),m.length<2){a&&(a.innerHTML="",a.classList.add("hidden"));return}c=setTimeout(async()=>{try{let v=(await ProductsApi.search({q:m,limit:5})).data||[];a&&v.length>0&&(a.innerHTML=`
                            <div class="py-2">
                                ${v.map(P=>`
                                    <a href="/products/${P.slug}/" class="flex items-center gap-3 px-4 py-2 hover:bg-gray-50">
                                        ${P.image?`<img src="${P.image}" alt="" class="w-10 h-10 object-cover rounded" onerror="this.style.display='none'">`:'<div class="w-10 h-10 bg-gray-100 rounded flex items-center justify-center"><svg class="w-5 h-5 text-gray-400" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="1.5" d="M4 16l4.586-4.586a2 2 0 012.828 0L16 16m-2-2l1.586-1.586a2 2 0 012.828 0L20 14m-6-6h.01M6 20h12a2 2 0 002-2V6a2 2 0 00-2-2H6a2 2 0 00-2 2v12a2 2 0 002 2z"/></svg></div>'}
                                        <div>
                                            <p class="text-sm font-medium text-gray-900">${Templates.escapeHtml(P.name)}</p>
                                            <p class="text-sm text-primary-600">${Templates.formatPrice(P.current_price??P.price_converted??P.price)}</p>
                                        </div>
                                    </a>
                                `).join("")}
                            </div>
                        `,a.classList.remove("hidden"))}catch(b){console.error("Search suggestions failed:",b)}},300)}),i?.addEventListener("blur",()=>{setTimeout(()=>{a&&a.classList.add("hidden")},200)})}async function E(){await L()}function H(y,i){let a=document.getElementById("search-results");if(!a)return;if(y.length===0){a.innerHTML=`
                <div class="text-center py-16">
                    <svg class="w-24 h-24 text-gray-300 mx-auto mb-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path stroke-linecap="round" stroke-linejoin="round" stroke-width="1.5" d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z"/>
                    </svg>
                    <h2 class="text-2xl font-bold text-gray-900 mb-2">No results found</h2>
                    <p class="text-gray-600 mb-4">We couldn't find any products matching "${Templates.escapeHtml(n)}"</p>
                    <p class="text-gray-500 text-sm">Try different keywords or browse our categories</p>
                </div>
            `;return}let c=Storage.get("productViewMode")||"grid",u=c==="list"?"space-y-4":"grid grid-cols-2 md:grid-cols-3 lg:grid-cols-4 gap-4 md:gap-6";a.innerHTML=`
            <div class="${u}">
                ${y.map(b=>ProductCard.render(b,{layout:c})).join("")}
            </div>
            ${i.total_pages>1?`
                <div id="search-pagination" class="mt-8">${Pagination.render({currentPage:i.current_page||e,totalPages:i.total_pages,totalItems:i.total})}</div>
            `:""}
        `,ProductCard.bindEvents(a),document.getElementById("search-pagination")?.addEventListener("click",b=>{let v=b.target.closest("[data-page]");v&&(e=parseInt(v.dataset.page),S(),E(),window.scrollTo({top:0,behavior:"smooth"}))})}async function U(){let y=document.getElementById("filter-categories");if(y)try{let a=(await CategoriesAPI.getAll({has_products:!0,limit:20})).data||[];y.innerHTML=`
                <div class="space-y-2">
                    <label class="flex items-center">
                        <input type="radio" name="category" value="" ${s.category?"":"checked"} class="text-primary-600 focus:ring-primary-500">
                        <span class="ml-2 text-sm text-gray-600">All Categories</span>
                    </label>
                    ${a.map(c=>`
                        <label class="flex items-center">
                            <input type="radio" name="category" value="${c.id}" ${s.category===String(c.id)?"checked":""} class="text-primary-600 focus:ring-primary-500">
                            <span class="ml-2 text-sm text-gray-600">${Templates.escapeHtml(c.name)}</span>
                        </label>
                    `).join("")}
                </div>
            `,B()}catch{}}function B(){document.querySelectorAll('input[name="category"]').forEach(i=>{i.addEventListener("change",a=>{a.target.value?s.category=a.target.value:delete s.category,e=1,S(),E()})})}function C(){let y=document.getElementById("apply-price-filter"),i=document.getElementById("filter-in-stock"),a=document.getElementById("clear-filters");y?.addEventListener("click",()=>{let m=document.getElementById("filter-min-price")?.value,b=document.getElementById("filter-max-price")?.value;m?s.min_price=m:delete s.min_price,b?s.max_price=b:delete s.max_price,e=1,S(),E()}),i?.addEventListener("change",m=>{m.target.checked?s.in_stock=!0:delete s.in_stock,e=1,S(),E()}),a?.addEventListener("click",()=>{s={},e=1,document.querySelectorAll('input[name="category"]').forEach(v=>{v.checked=v.value===""});let m=document.getElementById("filter-min-price"),b=document.getElementById("filter-max-price");m&&(m.value=""),b&&(b.value=""),i&&(i.checked=!1),S(),E()});let c=document.getElementById("filter-min-price"),u=document.getElementById("filter-max-price");c&&s.min_price&&(c.value=s.min_price),u&&s.max_price&&(u.value=s.max_price),i&&s.in_stock&&(i.checked=!0)}function k(){let y=document.getElementById("sort-select");y&&(y.value=s.ordering||"",y.addEventListener("change",i=>{i.target.value?s.ordering=i.target.value:delete s.ordering,e=1,S(),E()}))}function S(){let y=new URLSearchParams;n&&y.set("q",n),s.category&&y.set("category",s.category),s.min_price&&y.set("min_price",s.min_price),s.max_price&&y.set("max_price",s.max_price),s.ordering&&y.set("ordering",s.ordering),s.in_stock&&y.set("in_stock","true"),e>1&&y.set("page",e);let i=`${window.location.pathname}?${y.toString()}`;window.history.pushState({},"",i)}function T(){o&&(o.abort(),o=null),n="",e=1,s={},h=!1}return{init:w,destroy:T}})();window.SearchPage=xr;Gr=xr});var Er={};te(Er,{default:()=>Qr});var kr,Qr,Cr=ee(()=>{kr=(function(){"use strict";let n=1,e="added_desc",s="all";function o(){return window.BUNORAA_CURRENCY?.symbol||"\u09F3"}let h={get CURRENCY_SYMBOL(){return o()},PRIORITY_LEVELS:{low:{label:"Low",color:"gray",icon:"\u25CB"},normal:{label:"Normal",color:"blue",icon:"\u25D0"},high:{label:"High",color:"amber",icon:"\u25CF"},urgent:{label:"Urgent",color:"red",icon:"\u2605"}}};async function w(){AuthGuard.protectPage()&&(await A(),z())}function _(C={}){let k=C.product||C||{},S=[C.product_image,k.product_image,k.primary_image,k.image,Array.isArray(k.images)?k.images[0]:null,k.image_url,k.thumbnail],T=y=>{if(!y)return"";if(typeof y=="string")return y;if(typeof y=="object"){if(typeof y.image=="string"&&y.image)return y.image;if(y.image&&typeof y.image=="object"){if(typeof y.image.url=="string"&&y.image.url)return y.image.url;if(typeof y.image.src=="string"&&y.image.src)return y.image.src}if(typeof y.url=="string"&&y.url)return y.url;if(typeof y.src=="string"&&y.src)return y.src}return""};for(let y of S){let i=T(y);if(i)return i}return""}function L(C={}){let k=C.product||C||{},S=v=>{if(v==null)return null;let P=Number(v);return Number.isFinite(P)?P:null},T=[C.product_price,k.price,C.price,C.current_price,C.price_at_add],y=null;for(let v of T)if(y=S(v),y!==null)break;let i=[C.product_sale_price,k.sale_price,C.sale_price],a=null;for(let v of i)if(a=S(v),a!==null)break;let c=S(C.lowest_price_seen),u=S(C.highest_price_seen),m=S(C.target_price),b=S(C.price_at_add);return{price:y!==null?y:0,salePrice:a!==null?a:null,lowestPrice:c,highestPrice:u,targetPrice:m,priceAtAdd:b}}async function A(){let C=document.getElementById("wishlist-container");if(C){Loader.show(C,"skeleton");try{let k=await WishlistApi.getWishlist({page:n,sort:e}),S=[],T={};Array.isArray(k)?S=k:k&&typeof k=="object"&&(S=k.data||k.results||k.items||[],!Array.isArray(S)&&k.data&&typeof k.data=="object"?(S=k.data.items||k.data.results||[],T=k.data.meta||k.meta||{}):T=k.meta||{}),Array.isArray(S)||(S=S&&typeof S=="object"?[S]:[]);let y=S;s==="on_sale"?y=S.filter(i=>{let a=L(i);return a.salePrice&&a.salePrice<a.price}):s==="in_stock"?y=S.filter(i=>i.is_in_stock!==!1):s==="price_drop"?y=S.filter(i=>{let a=L(i);return a.priceAtAdd&&a.price<a.priceAtAdd}):s==="at_target"&&(y=S.filter(i=>{let a=L(i);return a.targetPrice&&a.price<=a.targetPrice})),Y(y,S,T)}catch(k){let S=k&&(k.message||k.detail)||"Failed to load wishlist.";if(k&&k.status===401){AuthGuard.redirectToLogin();return}C.innerHTML=`<p class="text-red-500 text-center py-8">${Templates.escapeHtml(S)}</p>`}}}function Y(C,k,S){let T=document.getElementById("wishlist-container");if(!T)return;let y=k.length,i=k.filter(u=>{let m=L(u);return m.salePrice&&m.salePrice<m.price}).length,a=k.filter(u=>{let m=L(u);return m.priceAtAdd&&m.price<m.priceAtAdd}).length,c=k.filter(u=>{let m=L(u);return m.targetPrice&&m.price<=m.targetPrice}).length;if(y===0){T.innerHTML=`
                <div class="text-center py-16">
                    <svg class="w-24 h-24 text-gray-300 mx-auto mb-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path stroke-linecap="round" stroke-linejoin="round" stroke-width="1.5" d="M4.318 6.318a4.5 4.5 0 000 6.364L12 20.364l7.682-7.682a4.5 4.5 0 00-6.364-6.364L12 7.636l-1.318-1.318a4.5 4.5 0 00-6.364 0z"/>
                    </svg>
                    <h2 class="text-2xl font-bold text-gray-900 dark:text-white mb-2">Your wishlist is empty</h2>
                    <p class="text-gray-600 dark:text-gray-400 mb-8">Start adding items you love to your wishlist.</p>
                    <a href="/products/" class="inline-flex items-center px-6 py-3 bg-primary-600 text-white font-semibold rounded-lg hover:bg-primary-700 transition-colors">
                        Browse Products
                    </a>
                </div>
            `;return}if(T.innerHTML=`
            <!-- Header with Stats -->
            <div class="mb-6">
                <div class="flex flex-col md:flex-row md:items-center md:justify-between gap-4">
                    <div>
                        <h1 class="text-2xl font-bold text-gray-900 dark:text-white">My Wishlist</h1>
                        <p class="text-sm text-gray-500 dark:text-gray-400 mt-1">${y} items saved</p>
                    </div>
                    <div class="flex flex-wrap gap-2">
                        <button id="add-all-to-cart-btn" class="px-4 py-2 bg-primary-600 text-white text-sm font-medium rounded-lg hover:bg-primary-700 transition-colors flex items-center">
                            <svg class="w-4 h-4 mr-2" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M3 3h2l.4 2M7 13h10l4-8H5.4M7 13L5.4 5M7 13l-2.293 2.293c-.63.63-.184 1.707.707 1.707H17m0 0a2 2 0 100 4 2 2 0 000-4zm-8 2a2 2 0 11-4 0 2 2 0 014 0z"></path>
                            </svg>
                            Add All to Cart
                        </button>
                        <button id="share-wishlist-btn" class="px-4 py-2 border border-gray-300 dark:border-gray-600 text-gray-700 dark:text-gray-300 text-sm font-medium rounded-lg hover:bg-gray-50 dark:hover:bg-gray-800 transition-colors flex items-center">
                            <svg class="w-4 h-4 mr-2" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M8.684 13.342C8.886 12.938 9 12.482 9 12c0-.482-.114-.938-.316-1.342m0 2.684a3 3 0 110-2.684m0 2.684l6.632 3.316m-6.632-6l6.632-3.316m0 0a3 3 0 105.367-2.684 3 3 0 00-5.367 2.684zm0 9.316a3 3 0 105.368 2.684 3 3 0 00-5.368-2.684z"></path>
                            </svg>
                            Share
                        </button>
                        <button id="clear-wishlist-btn" class="px-4 py-2 text-red-600 dark:text-red-400 text-sm font-medium hover:bg-red-50 dark:hover:bg-red-900/20 rounded-lg transition-colors">
                            Clear All
                        </button>
                    </div>
                </div>
                
                <!-- Quick Stats -->
                <div class="mt-4 grid grid-cols-2 md:grid-cols-4 gap-3">
                    <div class="bg-white dark:bg-gray-800 rounded-lg p-3 border border-gray-200 dark:border-gray-700">
                        <div class="text-2xl font-bold text-gray-900 dark:text-white">${y}</div>
                        <div class="text-xs text-gray-500 dark:text-gray-400">Total Items</div>
                    </div>
                    <div class="bg-white dark:bg-gray-800 rounded-lg p-3 border border-gray-200 dark:border-gray-700 ${i>0?"ring-2 ring-green-500":""}">
                        <div class="text-2xl font-bold text-green-600 dark:text-green-400">${i}</div>
                        <div class="text-xs text-gray-500 dark:text-gray-400">On Sale</div>
                    </div>
                    <div class="bg-white dark:bg-gray-800 rounded-lg p-3 border border-gray-200 dark:border-gray-700 ${a>0?"ring-2 ring-blue-500":""}">
                        <div class="text-2xl font-bold text-blue-600 dark:text-blue-400">${a}</div>
                        <div class="text-xs text-gray-500 dark:text-gray-400">Price Dropped</div>
                    </div>
                    <div class="bg-white dark:bg-gray-800 rounded-lg p-3 border border-gray-200 dark:border-gray-700 ${c>0?"ring-2 ring-amber-500":""}">
                        <div class="text-2xl font-bold text-amber-600 dark:text-amber-400">${c}</div>
                        <div class="text-xs text-gray-500 dark:text-gray-400">At Target Price</div>
                    </div>
                </div>
            </div>
            
            <!-- Filters and Sort -->
            <div class="mb-6 flex flex-col sm:flex-row sm:items-center sm:justify-between gap-4 bg-white dark:bg-gray-800 p-4 rounded-lg border border-gray-200 dark:border-gray-700">
                <div class="flex flex-wrap gap-2">
                    <button class="filter-btn px-3 py-1.5 text-sm rounded-lg transition-colors ${s==="all"?"bg-primary-600 text-white":"bg-gray-100 dark:bg-gray-700 text-gray-700 dark:text-gray-300"}" data-filter="all">All</button>
                    <button class="filter-btn px-3 py-1.5 text-sm rounded-lg transition-colors ${s==="on_sale"?"bg-primary-600 text-white":"bg-gray-100 dark:bg-gray-700 text-gray-700 dark:text-gray-300"}" data-filter="on_sale">On Sale</button>
                    <button class="filter-btn px-3 py-1.5 text-sm rounded-lg transition-colors ${s==="in_stock"?"bg-primary-600 text-white":"bg-gray-100 dark:bg-gray-700 text-gray-700 dark:text-gray-300"}" data-filter="in_stock">In Stock</button>
                    <button class="filter-btn px-3 py-1.5 text-sm rounded-lg transition-colors ${s==="price_drop"?"bg-primary-600 text-white":"bg-gray-100 dark:bg-gray-700 text-gray-700 dark:text-gray-300"}" data-filter="price_drop">Price Drop</button>
                    <button class="filter-btn px-3 py-1.5 text-sm rounded-lg transition-colors ${s==="at_target"?"bg-primary-600 text-white":"bg-gray-100 dark:bg-gray-700 text-gray-700 dark:text-gray-300"}" data-filter="at_target">At Target</button>
                </div>
                <div class="flex items-center gap-2">
                    <label class="text-sm text-gray-500 dark:text-gray-400">Sort:</label>
                    <select id="wishlist-sort" class="text-sm border-gray-300 dark:border-gray-600 rounded-lg bg-white dark:bg-gray-700 text-gray-700 dark:text-gray-300 focus:ring-primary-500 focus:border-primary-500">
                        <option value="added_desc" ${e==="added_desc"?"selected":""}>Newest First</option>
                        <option value="added_asc" ${e==="added_asc"?"selected":""}>Oldest First</option>
                        <option value="price_asc" ${e==="price_asc"?"selected":""}>Price: Low to High</option>
                        <option value="price_desc" ${e==="price_desc"?"selected":""}>Price: High to Low</option>
                        <option value="priority" ${e==="priority"?"selected":""}>Priority</option>
                        <option value="name" ${e==="name"?"selected":""}>Name A-Z</option>
                    </select>
                </div>
            </div>
            
            <!-- Items Grid -->
            <div class="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-6">
                ${C.map(u=>Z(u)).join("")}
            </div>
            
            ${C.length===0&&y>0?`
                <div class="text-center py-12">
                    <p class="text-gray-500 dark:text-gray-400">No items match the selected filter.</p>
                    <button class="mt-4 text-primary-600 hover:underline" onclick="document.querySelector('[data-filter=all]').click()">Show all items</button>
                </div>
            `:""}
            
            ${S.total_pages>1?'<div id="wishlist-pagination" class="mt-8"></div>':""}
        `,S&&S.total_pages>1){let u=document.getElementById("wishlist-pagination");if(u&&window.Pagination){let m=new window.Pagination({totalPages:S.total_pages,currentPage:S.current_page||n,className:"justify-center",onChange:b=>{n=b,A(),window.scrollTo({top:0,behavior:"smooth"})}});u.innerHTML="",u.appendChild(m.create())}}E()}function Z(C){try{let k=C.product||C||{},S=C.product_name||k.name||"",T=C.product_slug||k.slug||"",y=C.is_in_stock!==void 0?C.is_in_stock:k.is_in_stock!==void 0?k.is_in_stock:k.stock_quantity>0,i=_(C||{}),a=!!C.product_has_variants,c={price:0,salePrice:null,lowestPrice:null,highestPrice:null,targetPrice:null,priceAtAdd:null};try{c=L(C||{})}catch{c={price:0,salePrice:null}}let{price:u,salePrice:m,lowestPrice:b,highestPrice:v,targetPrice:P,priceAtAdd:D}=c,J=m||u,le=D&&J<D,be=D?Math.round((J-D)/D*100):0,ke=P&&J<=P,fe=m&&m<u,we=C.priority||"normal",ye=h.PRIORITY_LEVELS[we]||h.PRIORITY_LEVELS.normal,ce=me=>{try{return Templates.escapeHtml(me||"")}catch{return String(me||"")}},xe=me=>{try{return Price.render({price:me.price,salePrice:me.salePrice})}catch{return`<span class="font-bold">${h.CURRENCY_SYMBOL}${me.price||0}</span>`}},pe=me=>{try{return Templates.formatPrice(me)}catch{return`${h.CURRENCY_SYMBOL}${me}`}},Ee=k&&k.aspect&&(k.aspect.css||(k.aspect.width&&k.aspect.height?`${k.aspect.width}/${k.aspect.height}`:null))||"1/1";return`
                <div class="wishlist-item relative bg-white dark:bg-gray-800 rounded-xl shadow-sm border border-gray-100 dark:border-gray-700 overflow-hidden group" 
                     data-item-id="${C&&C.id?C.id:""}" 
                     data-product-id="${k&&k.id?k.id:C&&C.product?C.product:""}" 
                     data-product-slug="${ce(T)}" 
                     data-product-has-variants="${a}"
                     data-priority="${we}">
                    
                    <!-- Image Section -->
                    <div class="relative" style="aspect-ratio: ${Ee};">
                        <!-- Badges -->
                        <div class="absolute top-2 left-2 z-10 flex flex-col gap-1">
                            ${fe?`
                                <div class="bg-red-500 text-white text-xs font-bold px-2 py-1 rounded">
                                    -${Math.round((1-m/u)*100)}%
                                </div>
                            `:""}
                            ${le?`
                                <div class="bg-blue-500 text-white text-xs font-bold px-2 py-1 rounded flex items-center">
                                    <svg class="w-3 h-3 mr-1" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                        <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M19 14l-7 7m0 0l-7-7m7 7V3"></path>
                                    </svg>
                                    ${Math.abs(be)}% drop
                                </div>
                            `:""}
                            ${ke?`
                                <div class="bg-amber-500 text-white text-xs font-bold px-2 py-1 rounded flex items-center">
                                    <svg class="w-3 h-3 mr-1" fill="currentColor" viewBox="0 0 20 20">
                                        <path d="M9.049 2.927c.3-.921 1.603-.921 1.902 0l1.07 3.292a1 1 0 00.95.69h3.462c.969 0 1.371 1.24.588 1.81l-2.8 2.034a1 1 0 00-.364 1.118l1.07 3.292c.3.921-.755 1.688-1.54 1.118l-2.8-2.034a1 1 0 00-1.175 0l-2.8 2.034c-.784.57-1.838-.197-1.539-1.118l1.07-3.292a1 1 0 00-.364-1.118L2.98 8.72c-.783-.57-.38-1.81.588-1.81h3.461a1 1 0 00.951-.69l1.07-3.292z"></path>
                                    </svg>
                                    Target!
                                </div>
                            `:""}
                            ${y?"":`
                                <div class="bg-gray-800 text-white text-xs font-bold px-2 py-1 rounded">
                                    Out of Stock
                                </div>
                            `}
                        </div>
                        
                        <!-- Priority Indicator -->
                        <div class="absolute top-2 right-12 z-10">
                            <button class="priority-btn w-8 h-8 rounded-full bg-white dark:bg-gray-700 shadow-md flex items-center justify-center text-${ye.color}-500 hover:scale-110 transition-transform" title="Priority: ${ye.label}" data-item-id="${C.id}">
                                <span class="text-sm">${ye.icon}</span>
                            </button>
                        </div>
                        
                        <!-- Remove Button -->
                        <button class="remove-btn absolute top-2 right-2 z-20 w-8 h-8 bg-gray-900/80 text-white rounded-full shadow-lg flex items-center justify-center hover:bg-red-600 transition-colors opacity-0 group-hover:opacity-100" aria-label="Remove from wishlist">
                            <svg class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M6 18L18 6M6 6l12 12"/>
                            </svg>
                        </button>
                        
                        <!-- Product Image -->
                        <a href="/products/${ce(T)}/">
                            ${i?`
                                <img 
                                    src="${i}" 
                                    alt="${ce(S)}"
                                    class="w-full h-full object-cover group-hover:scale-105 transition-transform duration-300"
                                    loading="lazy"
                                >
                            `:`
                                <div class="w-full h-full bg-gray-200 dark:bg-gray-700 flex items-center justify-center text-gray-400 dark:text-gray-500 text-xs uppercase tracking-wide">No Image</div>
                            `}
                        </a>
                    </div>
                    
                    <!-- Content Section -->
                    <div class="p-4">
                        ${k&&k.category?`
                            <a href="/categories/${ce(k.category.slug)}/" class="text-xs text-gray-500 dark:text-gray-400 hover:text-primary-600 dark:hover:text-primary-400">
                                ${ce(k.category.name)}
                            </a>
                        `:""}
                        <h3 class="font-medium text-gray-900 dark:text-white mt-1 line-clamp-2">
                            <a href="/products/${ce(T)}/" class="hover:text-primary-600 dark:hover:text-primary-400">
                                ${ce(S)}
                            </a>
                        </h3>
                        
                        <!-- Price Section -->
                        <div class="mt-2">
                            ${xe({price:u,salePrice:m})}
                        </div>
                        
                        <!-- Price History -->
                        ${b||P?`
                            <div class="mt-2 text-xs space-y-1">
                                ${b?`
                                    <div class="flex items-center justify-between text-gray-500 dark:text-gray-400">
                                        <span>Lowest:</span>
                                        <span class="font-medium text-green-600 dark:text-green-400">${pe(b)}</span>
                                    </div>
                                `:""}
                                ${P?`
                                    <div class="flex items-center justify-between text-gray-500 dark:text-gray-400">
                                        <span>Target:</span>
                                        <span class="font-medium text-amber-600 dark:text-amber-400">${pe(P)}</span>
                                    </div>
                                `:""}
                            </div>
                        `:""}
                        
                        <!-- Rating -->
                        ${k&&k.average_rating?`
                            <div class="flex items-center gap-1 mt-2">
                                ${Templates.renderStars(k.average_rating)}
                                <span class="text-xs text-gray-500 dark:text-gray-400">(${k.review_count||0})</span>
                            </div>
                        `:""}
                        
                        <!-- Actions -->
                        <div class="mt-4 flex gap-2">
                            <button 
                                class="add-to-cart-btn flex-1 px-3 py-2 bg-primary-600 text-white font-medium rounded-lg hover:bg-primary-700 disabled:bg-gray-400 disabled:cursor-not-allowed transition-colors text-sm flex items-center justify-center"
                                ${y?"":"disabled"}
                            >
                                <svg class="w-4 h-4 mr-1" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                    <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M3 3h2l.4 2M7 13h10l4-8H5.4M7 13L5.4 5M7 13l-2.293 2.293c-.63.63-.184 1.707.707 1.707H17m0 0a2 2 0 100 4 2 2 0 000-4zm-8 2a2 2 0 11-4 0 2 2 0 014 0z"></path>
                                </svg>
                                ${a?"Options":y?"Add":"Sold Out"}
                            </button>
                            <button class="set-target-btn px-3 py-2 border border-gray-300 dark:border-gray-600 text-gray-700 dark:text-gray-300 rounded-lg hover:bg-gray-50 dark:hover:bg-gray-700 transition-colors text-sm" title="Set target price" data-item-id="${C.id}" data-current-price="${J}">
                                <svg class="w-4 h-4" fill="${P?"currentColor":"none"}" stroke="currentColor" viewBox="0 0 24 24">
                                    <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M15 17h5l-1.405-1.405A2.032 2.032 0 0118 14.158V11a6.002 6.002 0 00-4-5.659V5a2 2 0 10-4 0v.341C7.67 6.165 6 8.388 6 11v3.159c0 .538-.214 1.055-.595 1.436L4 17h5m6 0v1a3 3 0 11-6 0v-1m6 0H9"></path>
                                </svg>
                            </button>
                        </div>
                    </div>
                    
                    <!-- Added Date -->
                    ${C&&C.added_at?`
                        <div class="px-4 pb-3 border-t border-gray-100 dark:border-gray-700 pt-3">
                            <p class="text-xs text-gray-400 dark:text-gray-500">Added ${Templates.formatDate(C.added_at)}</p>
                        </div>
                    `:""}
                </div>
            `}catch(k){return console.error("Failed to render wishlist item:",k),'<div class="p-4 bg-white dark:bg-gray-800 rounded shadow text-gray-500 dark:text-gray-400">Failed to render item</div>'}}function z(){document.getElementById("wishlist-sort")?.addEventListener("change",C=>{e=C.target.value,A()}),document.querySelectorAll(".filter-btn").forEach(C=>{C.addEventListener("click",()=>{s=C.dataset.filter,A()})})}function E(){let C=document.getElementById("clear-wishlist-btn"),k=document.getElementById("add-all-to-cart-btn"),S=document.getElementById("share-wishlist-btn"),T=document.querySelectorAll(".wishlist-item"),y=document.getElementById("wishlist-sort"),i=document.querySelectorAll(".filter-btn");y?.addEventListener("change",a=>{e=a.target.value,A()}),i.forEach(a=>{a.addEventListener("click",()=>{s=a.dataset.filter,A()})}),C?.addEventListener("click",async()=>{if(await Modal.confirm({title:"Clear Wishlist",message:"Are you sure you want to remove all items from your wishlist?",confirmText:"Clear All",cancelText:"Cancel"}))try{await WishlistApi.clear(),Toast.success("Wishlist cleared."),await A()}catch(c){Toast.error(c.message||"Failed to clear wishlist.")}}),k?.addEventListener("click",async()=>{let a=k;a.disabled=!0,a.innerHTML='<svg class="animate-spin w-4 h-4 mr-2" fill="none" viewBox="0 0 24 24"><circle class="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" stroke-width="4"></circle><path class="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path></svg>Adding...';try{let c=document.querySelectorAll('.wishlist-item:not([data-product-has-variants="true"])'),u=0,m=0;for(let b of c){let v=b.dataset.productId;if(v)try{await CartApi.addItem(v,1),u++}catch{m++}}u>0&&(Toast.success(`Added ${u} items to cart!`),document.dispatchEvent(new CustomEvent("cart:updated"))),m>0&&Toast.warning(`${m} items could not be added (may require variant selection).`)}catch(c){Toast.error(c.message||"Failed to add items to cart.")}finally{a.disabled=!1,a.innerHTML='<svg class="w-4 h-4 mr-2" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M3 3h2l.4 2M7 13h10l4-8H5.4M7 13L5.4 5M7 13l-2.293 2.293c-.63.63-.184 1.707.707 1.707H17m0 0a2 2 0 100 4 2 2 0 000-4zm-8 2a2 2 0 11-4 0 2 2 0 014 0z"></path></svg>Add All to Cart'}}),S?.addEventListener("click",async()=>{try{let a=`${window.location.origin}/wishlist/share/`;navigator.share?await navigator.share({title:"My Wishlist",text:"Check out my wishlist!",url:a}):(await navigator.clipboard.writeText(a),Toast.success("Wishlist link copied to clipboard!"))}catch(a){a.name!=="AbortError"&&Toast.error("Failed to share wishlist.")}}),T.forEach(a=>{let c=a.dataset.itemId,u=a.dataset.productId,m=a.dataset.productSlug;a.querySelector(".remove-btn")?.addEventListener("click",async()=>{try{await WishlistApi.removeItem(c),Toast.success("Removed from wishlist."),a.remove(),document.querySelectorAll(".wishlist-item").length===0&&await A()}catch(b){Toast.error(b.message||"Failed to remove item.")}}),a.querySelector(".priority-btn")?.addEventListener("click",async()=>{let b=["low","normal","high","urgent"],v=a.dataset.priority||"normal",P=b.indexOf(v),D=b[(P+1)%b.length];try{WishlistApi.updateItem&&await WishlistApi.updateItem(c,{priority:D}),a.dataset.priority=D;let J=a.querySelector(".priority-btn"),le=h.PRIORITY_LEVELS[D];J.title=`Priority: ${le.label}`,J.innerHTML=`<span class="text-sm">${le.icon}</span>`,J.className=`priority-btn w-8 h-8 rounded-full bg-white dark:bg-gray-700 shadow-md flex items-center justify-center text-${le.color}-500 hover:scale-110 transition-transform`,Toast.success(`Priority set to ${le.label}`)}catch{Toast.error("Failed to update priority.")}}),a.querySelector(".set-target-btn")?.addEventListener("click",async()=>{let b=parseFloat(a.querySelector(".set-target-btn").dataset.currentPrice)||0,v=`
                    <div class="space-y-4">
                        <p class="text-sm text-gray-600 dark:text-gray-400">Set a target price and we'll notify you when the item drops to or below this price.</p>
                        <div>
                            <label class="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">Current Price</label>
                            <div class="text-lg font-bold text-gray-900 dark:text-white">${h.CURRENCY_SYMBOL}${b.toLocaleString()}</div>
                        </div>
                        <div>
                            <label class="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">Target Price</label>
                            <div class="flex items-center">
                                <span class="text-gray-500 mr-2">${h.CURRENCY_SYMBOL}</span>
                                <input type="number" id="target-price-input" value="${Math.round(b*.9)}" min="1" max="${b}" class="flex-1 px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-lg bg-white dark:bg-gray-700 text-gray-900 dark:text-white focus:ring-primary-500 focus:border-primary-500">
                            </div>
                            <p class="text-xs text-gray-500 dark:text-gray-400 mt-1">Suggested: ${h.CURRENCY_SYMBOL}${Math.round(b*.9).toLocaleString()} (10% off)</p>
                        </div>
                    </div>
                `,P=await Modal.open({title:"Set Target Price",content:v,confirmText:"Set Alert",cancelText:"Cancel",onConfirm:async()=>{let D=parseFloat(document.getElementById("target-price-input").value);if(!D||D<=0)return Toast.error("Please enter a valid target price."),!1;try{return WishlistApi.updateItem&&await WishlistApi.updateItem(c,{target_price:D}),Toast.success(`Price alert set for ${h.CURRENCY_SYMBOL}${D.toLocaleString()}`),await A(),!0}catch{return Toast.error("Failed to set price alert."),!1}}})}),a.querySelector(".add-to-cart-btn")?.addEventListener("click",async b=>{let v=b.target.closest(".add-to-cart-btn");if(v.disabled)return;v.disabled=!0;let P=v.innerHTML;if(v.innerHTML='<svg class="animate-spin w-4 h-4" fill="none" viewBox="0 0 24 24"><circle class="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" stroke-width="4"></circle><path class="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path></svg>',a.dataset.productHasVariants==="true"||a.dataset.productHasVariants==="True"||a.dataset.productHasVariants==="1"){U(a),v.disabled=!1,v.innerHTML=P;return}try{await CartApi.addItem(u,1),Toast.success("Added to cart!"),document.dispatchEvent(new CustomEvent("cart:updated"))}catch(J){if(!!(J&&(J.errors&&J.errors.variant_id||J.message&&typeof J.message=="string"&&J.message.toLowerCase().includes("variant")))&&(Toast.info("This product requires selecting a variant."),m)){window.location.href=`/products/${m}/`;return}Toast.error(J.message||"Failed to add to cart.")}finally{v.disabled=!1,v.innerHTML=P}})})}function H(C){let k={};return C.forEach(S=>{k[S.attribute_name]||(k[S.attribute_name]=[]),k[S.attribute_name].push(S)}),Object.entries(k).map(([S,T])=>`
            <div class="mt-4">
                <label class="text-sm font-medium text-gray-700">${Templates.escapeHtml(S)}:</label>
                <div class="flex flex-wrap gap-2 mt-2" id="wishlist-variant-group-${Templates.slugify(S)}">
                    ${T.map((y,i)=>`
                        <button type="button" class="wishlist-modal-variant-btn px-3 py-2 border rounded-lg text-sm transition-colors ${i===0?"border-primary-500 bg-primary-50 text-primary-700":"border-gray-300 hover:border-gray-400"}" data-variant-id="${y.id}" data-price="${y.price_converted??y.price??""}" data-stock="${y.stock_quantity||0}">
                            ${Templates.escapeHtml(y.value)}
                            ${y.price_converted??y.price?`<span class="text-xs text-gray-500"> (${Templates.formatPrice(y.price_converted??y.price)})</span>`:""}
                        </button>
                    `).join("")}
                </div>
            </div>
        `).join("")}async function U(C){let k=C.product_slug||C.dataset?.productSlug||"",S=C.product||C.dataset?.productId||"";try{let T;if(typeof ProductsApi<"u"&&ProductsApi.getProduct)T=await ProductsApi.getProduct(k||S);else{let m=window.BUNORAA_CURRENCY&&window.BUNORAA_CURRENCY.code||void 0;T=await ApiClient.get(`/catalog/products/${k||S}/`,{currency:m})}if(!T||!T.success||!T.data){let m=T&&T.message?T.message:"Failed to load product variants.";Toast.error(m);return}let y=T.data,i=y.variants||[];if(!i.length){window.location.href=`/products/${y.slug||k||S}/`;return}let a=y.images?.[0]?.image||y.primary_image||y.image||"",c=`
                <div class="grid grid-cols-1 md:grid-cols-3 gap-4">
                    <div class="col-span-1">
                        ${a?`<img src="${a}" class="w-full h-48 object-cover rounded" alt="${Templates.escapeHtml(y.name)}">`:'<div class="w-full h-48 bg-gray-100 rounded"></div>'}
                    </div>
                    <div class="col-span-2">
                        <h3 class="text-lg font-semibold">${Templates.escapeHtml(y.name)}</h3>
                        <div id="wishlist-variant-price" class="mt-2 text-lg font-bold">${Templates.formatPrice(i?.[0]?.price_converted??i?.[0]?.price??y.price)}</div>
                        <div id="wishlist-variant-options" class="mt-4">
                            ${H(i)}
                        </div>
                        <div class="mt-4 flex items-center gap-2">
                            <label class="text-sm text-gray-700">Qty</label>
                            <input id="wishlist-variant-qty" type="number" value="1" min="1" class="w-20 px-3 py-2 border rounded" />
                        </div>
                    </div>
                </div>
            `,u=await Modal.open({title:"Select Variant",content:c,confirmText:"Add to Cart",cancelText:"Cancel",size:"md",onConfirm:async()=>{let b=document.querySelector(".wishlist-modal-variant-btn.border-primary-500")||document.querySelector(".wishlist-modal-variant-btn");if(!b)return Toast.error("Please select a variant."),!1;let v=b.dataset.variantId,P=parseInt(document.getElementById("wishlist-variant-qty")?.value)||1;try{return await CartApi.addItem(y.id,P,v),Toast.success("Added to cart!"),document.dispatchEvent(new CustomEvent("cart:updated")),!0}catch(D){return Toast.error(D.message||"Failed to add to cart."),!1}}});setTimeout(()=>{let m=document.querySelectorAll(".wishlist-modal-variant-btn");m.forEach(v=>{v.addEventListener("click",()=>{m.forEach(D=>D.classList.remove("border-primary-500","bg-primary-50","text-primary-700")),v.classList.add("border-primary-500","bg-primary-50","text-primary-700");let P=v.dataset.price;if(P!==void 0){let D=document.getElementById("wishlist-variant-price");D&&(D.textContent=Templates.formatPrice(P))}})});let b=document.querySelector(".wishlist-modal-variant-btn");b&&b.click()},20)}catch{Toast.error("Failed to load variants.")}}function B(){n=1}return{init:w,destroy:B}})();window.WishlistPage=kr;Qr=kr});var Jr,xt=ee(()=>{Jr=Lt({"./pages/account.js":()=>Promise.resolve().then(()=>(Us(),Ds)),"./pages/cart.js":()=>Promise.resolve().then(()=>(Gs(),Ys)),"./pages/category.js":()=>Promise.resolve().then(()=>(Xs(),Js)),"./pages/checkout.js":()=>Promise.resolve().then(()=>(er(),Zs)),"./pages/contact.js":()=>Promise.resolve().then(()=>(rr(),sr)),"./pages/faq.js":()=>Promise.resolve().then(()=>(or(),nr)),"./pages/home.js":()=>Promise.resolve().then(()=>(ur(),cr)),"./pages/orders.js":()=>Promise.resolve().then(()=>(gr(),pr)),"./pages/preorders.js":()=>Promise.resolve().then(()=>Pr(hr())),"./pages/product.js":()=>Promise.resolve().then(()=>(br(),yr)),"./pages/search.js":()=>Promise.resolve().then(()=>(wr(),vr)),"./pages/wishlist.js":()=>Promise.resolve().then(()=>(Cr(),Er))})});var Xr=_t(()=>{Os();xt();var vt=(function(){"use strict";let n={},e=null,s=null;async function o(T){try{let y=await Jr(`./pages/${T}.js`);return y.default||y}catch(y){return console.warn(`Page controller for ${T} not found:`,y),null}}function h(){A(),Y(),Z(),z(),w(),U(),C(),k();try{let T=performance.getEntriesByType?performance.getEntriesByType("navigation"):[],y=T&&T[0]||null;y&&y.type==="navigate"&&!window.location.hash&&setTimeout(()=>{let i=document.scrollingElement||document.documentElement;if(!i)return;let a=i.scrollTop||window.pageYOffset||0,c=Math.max(0,i.scrollHeight-window.innerHeight);a>Math.max(100,c*.6)&&window.scrollTo({top:0,behavior:"auto"})},60)}catch{}}async function w(){try{if(!AuthApi.isAuthenticated()){let a=localStorage.getItem("wishlist");if(a){let c=JSON.parse(a);WishlistApi.updateBadge(c);let u=c.items||c.data&&c.data.items||[];L(u)}else L([]);return}let y=(await WishlistApi.getWishlist({pageSize:200})).data||{},i=y.items||y.data||[];WishlistApi.updateBadge(y),L(i)}catch{try{let y=localStorage.getItem("wishlist");if(y){let i=JSON.parse(y);WishlistApi.updateBadge(i);let a=i.items||i.data&&i.data.items||[];L(a)}}catch{}}}let _=[];function L(T){try{_=T||[];let y={},i={};(T||[]).forEach(a=>{let c=a.product||a.product_id||a.product&&a.product.id||null,u=a.product_slug||a.product&&a.product.slug||null,m=a.id||a.pk||a.uuid||a.item||null;c&&(y[String(c)]=m||!0),u&&(i[String(u)]=m||!0)}),document.querySelectorAll(".wishlist-btn").forEach(a=>{try{let c=a.querySelector("svg"),u=c?.querySelector(".heart-fill"),m=a.dataset.productId||a.closest("[data-product-id]")?.dataset.productId,b=a.dataset.productSlug||a.closest("[data-product-slug]")?.dataset.productSlug,v=null;m&&y.hasOwnProperty(String(m))?v=y[String(m)]:b&&i.hasOwnProperty(String(b))&&(v=i[String(b)]),v?(a.dataset.wishlistItemId=v,a.classList.add("text-red-500"),a.setAttribute("aria-pressed","true"),c?.classList.add("fill-current"),u&&(u.style.opacity="1")):(a.removeAttribute("data-wishlist-item-id"),a.classList.remove("text-red-500"),a.setAttribute("aria-pressed","false"),c?.classList.remove("fill-current"),u&&(u.style.opacity="0"))}catch{}})}catch{}}(function(){if(typeof MutationObserver>"u")return;let T=null;new MutationObserver(function(i){let a=!1;for(let c of i){if(c.addedNodes&&c.addedNodes.length){for(let u of c.addedNodes)if(u.nodeType===1&&(u.matches?.(".product-card")||u.querySelector?.(".product-card")||u.querySelector?.(".wishlist-btn"))){a=!0;break}}if(a)break}a&&(clearTimeout(T),T=setTimeout(()=>{try{L(_)}catch{}},150))}).observe(document.body,{childList:!0,subtree:!0})})();function A(){let T=window.location.pathname,y=document.body;if(y.dataset.page){e=y.dataset.page;return}if((T.startsWith("/accounts/")||T.startsWith("/account/"))&&!(T.startsWith("/accounts/profile")||T.startsWith("/account/profile"))){e=null;return}T==="/"||T==="/home/"?e="home":T==="/categories/"||T==="/products/"?e="search":T.startsWith("/categories/")&&T!=="/categories/"?e="category":T.startsWith("/products/")&&T!=="/products/"?e="product":T==="/search/"||T.startsWith("/search")?e="search":T.startsWith("/cart")?e="cart":T.startsWith("/checkout")?e="checkout":T==="/account"||T.startsWith("/account/")||T.startsWith("/accounts/profile")?e="account":T.startsWith("/orders")?e="orders":T.startsWith("/wishlist")?e="wishlist":T.startsWith("/contact")&&(e="contact")}function Y(){typeof Tabs<"u"&&document.querySelector("[data-tabs]")&&Tabs.init(),typeof Dropdown<"u"&&document.querySelectorAll("[data-dropdown-trigger]").forEach(T=>{let y=T.dataset.dropdownTarget,i=document.getElementById(y);i&&Dropdown.create(T,{content:i.innerHTML})});try{Rs()}catch{}}async function Z(){if(!e)return;try{s&&typeof s.destroy=="function"&&s.destroy()}catch{}let T=await o(e);if(T&&typeof T.init=="function"){s=T;try{await s.init()}catch(y){console.error("failed to init page controller",y)}}}try{"serviceWorker"in navigator&&navigator.serviceWorker.register("/static/js/sw.js").catch(()=>{})}catch{}async function z(){if(document.querySelectorAll("[data-cart-count]").length)try{let i=(await CartApi.getCart()).data?.item_count||0;try{localStorage.setItem("cart",JSON.stringify({item_count:i,savedAt:Date.now()}))}catch{}H(i)}catch{try{let i=localStorage.getItem("cart");if(i){let c=JSON.parse(i)?.item_count||0;H(c);return}}catch(i){console.error("Failed to get cart count fallback:",i)}}}async function E(T){try{return(((await WishlistApi.getWishlist({pageSize:200})).data||{}).items||[]).find(u=>String(u.product)===String(T))?.id||null}catch{return null}}function H(T){document.querySelectorAll("[data-cart-count]").forEach(i=>{i.textContent=T>99?"99+":T,i.classList.toggle("hidden",T===0)})}function U(){document.addEventListener("cart:updated",async()=>{await z()}),document.addEventListener("wishlist:updated",async()=>{await w()}),document.addEventListener("auth:login",()=>{B(!0)}),document.addEventListener("auth:logout",()=>{B(!1)}),document.querySelectorAll(".wishlist-btn").forEach(y=>{try{let i=y.querySelector("svg"),a=i?.querySelector(".heart-fill");y.classList.contains("text-red-500")?(i?.classList.add("fill-current"),a&&(a.style.opacity="1")):a&&(a.style.opacity="0")}catch{}}),document.addEventListener("click",async y=>{let i=y.target.closest("[data-quick-add], [data-add-to-cart], .add-to-cart-btn");if(i){y.preventDefault();let a=i.dataset.productId||i.dataset.quickAdd||i.dataset.addToCart;if(!a)return;i.disabled=!0;let c=i.innerHTML;i.innerHTML='<svg class="animate-spin h-4 w-4 mx-auto" fill="none" viewBox="0 0 24 24"><circle class="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" stroke-width="4"></circle><path class="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path></svg>';try{await CartApi.addItem(a,1),Toast.success("Added to cart!"),document.dispatchEvent(new CustomEvent("cart:updated"))}catch(u){Toast.error(u.message||"Failed to add to cart.")}finally{i.disabled=!1,i.innerHTML=c}}}),document.addEventListener("click",async y=>{let i=y.target.closest("[data-wishlist-toggle], .wishlist-btn");if(i){if(y.preventDefault(),!AuthApi.isAuthenticated()){Toast.warning("Please login to add items to your wishlist."),window.location.href="/account/login/?next="+encodeURIComponent(window.location.pathname);return}let a=i.dataset.wishlistToggle||i.dataset.productId||i.closest("[data-product-id]")?.dataset.productId;i.disabled=!0;let c=i.dataset.wishlistItemId||"";!c&&i.classList.contains("text-red-500")&&(c=await E(a)||"");let m=i.classList.contains("text-red-500")&&c;try{if(m){let b=await WishlistApi.removeItem(c);i.classList.remove("text-red-500"),i.setAttribute("aria-pressed","false"),i.querySelector("svg")?.classList.remove("fill-current");let v=i.querySelector("svg")?.querySelector(".heart-fill");v&&(v.style.opacity="0"),i.removeAttribute("data-wishlist-item-id"),Toast.success("Removed from wishlist.")}else{let b=await WishlistApi.addItem(a),v=b.data?.id||b.data?.item?.id||await E(a);v&&(i.dataset.wishlistItemId=v),i.classList.add("text-red-500"),i.setAttribute("aria-pressed","true"),i.querySelector("svg")?.classList.add("fill-current");let P=i.querySelector("svg")?.querySelector(".heart-fill");P&&(P.style.opacity="1"),Toast.success(b.message||"Added to wishlist!")}}catch(b){console.error("wishlist:error",b),Toast.error(b.message||"Failed to update wishlist.")}finally{i.disabled=!1}}}),document.addEventListener("click",y=>{let i=y.target.closest("[data-quick-view], .quick-view-btn");if(i){y.preventDefault();let a=i.dataset.quickView||i.dataset.productId,c=i.dataset.productSlug;c?window.location.href=`/products/${c}/`:a&&(typeof Modal<"u"&&Modal.showQuickView?Modal.showQuickView(a):window.location.href=`/products/${a}/`)}}),document.addEventListener("click",async y=>{if(y.target.closest("[data-logout]")){y.preventDefault();try{await AuthApi.logout(),Toast.success("Logged out successfully."),document.dispatchEvent(new CustomEvent("auth:logout")),window.location.href="/"}catch{Toast.error("Failed to logout.")}}});let T=document.getElementById("back-to-top");T&&(window.addEventListener("scroll",Debounce.throttle(()=>{window.scrollY>500?T.classList.remove("opacity-0","pointer-events-none"):T.classList.add("opacity-0","pointer-events-none")},100)),T.addEventListener("click",()=>{window.scrollTo({top:0,behavior:"smooth"})}))}function B(T){document.querySelectorAll("[data-auth-state]").forEach(i=>{let a=i.dataset.authState;a==="logged-in"?i.classList.toggle("hidden",!T):a==="logged-out"&&i.classList.toggle("hidden",T)})}function C(){let T=document.getElementById("mobile-menu-btn"),y=document.getElementById("close-mobile-menu"),i=document.getElementById("mobile-menu"),a=document.getElementById("mobile-menu-overlay");function c(){i?.classList.remove("translate-x-full"),a?.classList.remove("hidden"),document.body.classList.add("overflow-hidden")}function u(){i?.classList.add("translate-x-full"),a?.classList.add("hidden"),document.body.classList.remove("overflow-hidden")}T?.addEventListener("click",c),y?.addEventListener("click",u),a?.addEventListener("click",u)}function k(){let T=document.querySelector("[data-language-selector]"),y=document.getElementById("language-dropdown");T&&y&&(Dropdown.create(T,y),y.querySelectorAll("[data-language]").forEach(i=>{i.addEventListener("click",async()=>{let a=i.dataset.language;try{await LocalizationApi.setLanguage(a),Storage.set("language",a),window.location.reload()}catch{Toast.error("Failed to change language.")}})}))}function S(){s&&typeof s.destroy=="function"&&s.destroy(),e=null,s=null}return{init:h,destroy:S,getCurrentPage:()=>e,updateCartBadge:H}})();document.readyState==="loading"?document.addEventListener("DOMContentLoaded",vt.init):vt.init();window.App=vt});Xr();})();
//# sourceMappingURL=app.bundle.js.map
