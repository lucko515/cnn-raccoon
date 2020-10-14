function openAnalytics(tabId, menuItemId) {
    /*
        Function that handles UI interactions with side menu in the Dashboard.
     */
  let tabcontent = document.getElementsByClassName("tabcontent");
  let menu_elements = document.getElementsByClassName("tablinks");

  for (let i = 0; i < tabcontent.length; i++) {
    if (tabcontent[i].id === tabId){
      tabcontent[i].style.display = "block";
    }else{
      tabcontent[i].style.display = "none";
    }
  }

  for (let i = 0; i < menu_elements.length; i++) {
    if (menu_elements[i].id === menuItemId){
      menu_elements[i].classList.add("active-tab");
    }else{
      menu_elements[i].classList.remove("active-tab");
    }
  }
}

function imageSelector(evt){
    /*
        Helper function to create visual effect on the selected Image in the Dashboard.
     */
  let inputImages = document.getElementsByClassName("inputImgSelector");

  for (let i = 0; i < inputImages.length; i++) {
    let img = inputImages[i];
    if (inputImages[i].alt === evt.target.alt){
      img.classList.add("selected-img");
    }else{
      img.classList.remove("selected-img");
    }
  }

  selected_image_id = parseInt(evt.target.alt);
  calculateParams(selected_image_id, selected_layer_name);
}

function calculateParams(image_id, layer_id){
    /*
        Calculates model params (Feature maps, GradCam, Saliency maps) based on the selected layer and image.
     */
    document.getElementById("weights-container").innerHTML = "";
    document.getElementById("grad-cam-container").innerHTML = "";
    document.getElementById("saliency-map-container").innerHTML = "";

  // Weights showcase
  let init_weights = weights_paths[layer_id][image_id];
   for (let i = 0; i < init_weights.length; i++) {
       let template_clone = template_weights.content.cloneNode(true);
       template_clone.querySelector("#weights-img").src = "static/" + init_weights[i];
       weightsContainer.appendChild(template_clone);
   }

   // GradCam
   for (let i = 0; i < grad_cam_paths[image_id].length; i++) {
     let template_clone = template_gram_cam.content.cloneNode(true);
     template_clone.querySelector("#grad-cam-img").src = "static/" + grad_cam_paths[image_id][i];
     template_clone.querySelector("#img-desc").innerHTML = "Class - " + i.toString();

    gramCamContainer.appendChild(template_clone);
   }

   try{
       // Saliency Map
       let saliency_maps = saliency_map_paths[layer_id][image_id];

       let template_clone = template_saliency_map.content.cloneNode(true);
       template_clone.querySelector("#saliency-map-img").src = "static/" + saliency_maps;
       saliencyMapContainer.appendChild(template_clone);
   }
   catch (err){
       console.warn("Saliency map not calculated.");
   }

}
