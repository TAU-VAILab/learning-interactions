
function previous() {
    console.log(ww_active)
    if (ww_active) {
        if (counter_ww > 0) {
            counter_ww -= 1
            console.log(counter_ww)
        }
    } else {
        if (counter_is > 0) {
            counter_is -= 1
            console.log(counter_is)
        }
    }
    update_view()
}
function next() {
    console.log(ww_active)
    if (ww_active) {
        if (counter_ww + 1 < n_ww) {
            counter_ww += 1
            console.log(counter_ww)
        }
    } else {
        if (counter_is + 1 < n_is) {
            counter_is += 1
            console.log(counter_is)
        }
    }
    update_view()
}

function numchange() {
    var in_elt = document.getElementById("item_number")
    if (ww_active) {
        counter_ww = in_elt.value - 1
    } else {
        counter_is = in_elt.value - 1
    }
    update_view()
}

function switch_dataset() {
    var set = document.getElementById("dataset").value
    ww_active = set === "ww"
    console.log(set)
    console.log(ww_active)
    update_view()
}

function can_show_name(name) {
    return (name in ww_sources) & (ww_sources[name] === "Who's Waldo")
}

function name_to_elt(ds, name) {
    if(can_show_name(name)) {
        var img_src = "../assets/viz_ww/" + name + ".jpg"
        return '<img class="viz" src="' + img_src + '"></img>'
    } else {

        var urls_list = ds == "ww" ? ww_urls : is_urls

        var out = '<span class="cannot">Please view image at: '
        var key = name
        if(name in ww_sources & ww_sources[name] == "Conceptual Captions") {
            key = name.slice(0, -1)
        }
        if(key.startsWith("0") & (key.slice(1) in urls_list)) {
            key = key.slice(1)
        }
        if(key.startsWith("00") & (key.slice(2) in urls_list)) {
            key = key.slice(2)
        }
        if(key in urls_list) {
            var url = urls_list[key]
            out += '<a href="' + url + '" target="_blank">(link)</a>'
        } else {
            out += '(n/a)'
        }
        out += '</span>'
        return out
    }
}

function get_metrics(ds, model, name) {
    var permodel = metrics[ds][model]
    var vals = name in permodel ? permodel[name] : permodel[name.slice(0, -1)]

    if (ds === "ww") {
        var bl = vals['bl'].toFixed(2)
        var pe = vals['pe'].toFixed(2)
        var pc = vals['pc'].toFixed(2)
        var sim = vals['sim'].toFixed(2)
        var out = "BL " + bl + ", p<sub>e</sub> " + pe + ", p<sub>c</sub> " + pc + ", sim " + sim
    } else {
        var sim = vals['sim'].toFixed(2)
        var out = "sim " + sim
    }
    
    return out
}

function update_view() {
    var c = ww_active ? counter_ww : counter_is
    var c1 = 1 + c
    var c2 = ww_active ? n_ww : n_is
    // document.getElementById("c1").innerHTML = c1
    document.getElementById("c2").innerHTML = c2
    // document.getElementById("dsname").innerHTML = ww_active ? "Waldo and Wenda" : "imSitu-HHI"

    var in_elt = document.getElementById("item_number")
    in_elt.value = c1
    in_elt.min = 1
    in_elt.max = ww_active ? 1000 : 8021
    

    var model = document.getElementById("model").value
    var ds = ww_active ? "ww" : "is"

    var name = (ww_active ? ww_names : is_names)[c]

    var elt_img = document.getElementById("results_img")

    elt_img.innerHTML = name_to_elt(ds, name)
    // elt.innerHTML += '<br><br>'

    var elt = document.getElementById("results")
    elt.innerHTML = ''

    var mygt = gt[ds][name]

    if (ds === "is") {
        var source = is_sources[name]
        elt.innerHTML += '<b>ID: </b>' + name + ' (from imSitu ' + source + ' split*)<br>'
        elt.innerHTML += '<small>*Relevant for CoFormer which was trained on the imSitu train split</small><br><br>'
    } else {
        var source = ww_sources[name]
        var id = source === "Conceptual Captions" ? name.slice(0, -1) : name
        elt.innerHTML += '<b>ID: </b>' + id + ' (from ' + source + ')<br><br>'
    }
    elt.innerHTML += '<b>Ground truth: </b>' + mygt + '<br><br>'

    var res = (ww_active ? ww_res : is_res)[model][name]
    var top = res[0]

    console.log(res)

    elt.innerHTML += '<b>Top prediction:</b> ' + top + '<br><br>'
    elt.innerHTML += '<b>Metric(s):</b> ' + get_metrics(ds, model, name) + ' <br><br>'

    if (res.length > 1) {

        elt.innerHTML += '<b>Other predictions (lower-scoring beams):</b>'
        elt.innerHTML += '<ol>'
        for (var i = 1; i < res.length; i++) {
            elt.innerHTML += '<li>' + res[i] + '</li>'
        }
        elt.innerHTML += '</ol>'
    }

    // res.innerHTML = model + " " + ds + " " + c + " " + name + " " + img_src

}

update_view()