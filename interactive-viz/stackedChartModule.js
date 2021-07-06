var StackedChartModule = function(series, canvas_width, canvas_height) {
    // Create the tag:
    var canvas_tag = "<canvas width='" + canvas_width + "' height='" + canvas_height + "' ";
    canvas_tag += "style='border:1px dotted'></canvas>";
    // Append it to #elements
    var canvas = $(canvas_tag)[0];
    $("#elements").append(canvas);
    // Create the context and the drawing controller:
    var context = canvas.getContext("2d");

    var convertColorOpacity = function(hex) {

        if (hex.indexOf('#') != 0) {
            return 'rgba(0,0,0,0.1)';
        }

        hex = hex.replace('#', '');
        r = parseInt(hex.substring(0, 2), 16);
        g = parseInt(hex.substring(2, 4), 16);
        b = parseInt(hex.substring(4, 6), 16);
        return 'rgba(' + r + ',' + g + ',' + b + ',0.1)';
    };

    function shuffleArray(array) {
        for (let i = array.length - 1; i > 0; i--) {
            const j = Math.floor(Math.random() * (i + 1));
            [array[i], array[j]] = [array[j], array[i]];
        }
        return array
    }

    // For generating random (distinguishable) colours
    function selectColor(colorNum, colors){
        if (colors < 1) colors = 1; // defaults to one color - avoid divide by zero
        return "hsl(" + (colorNum * (360 / colors) % 360) + ",100%,50%)";
    }

    // Prep the chart properties and series:
    var datasets = []
    for (var i in series) {
        var s = series[i];
        var datasets_number = s.Num_agents;

        for (var color_indexes=[],i=0;i<datasets_number;++i) color_indexes[i]=i;
        color_indexes = shuffleArray(color_indexes)

        for (var j=0; j< datasets_number; j++) {
            var color = selectColor(color_indexes[j], datasets_number);
            var pool_nr = j + 1;
            var new_series = {
                label: s.Label + pool_nr,
                borderColor: color,
                backgroundColor: color,
                fill: true,
                data: []
            };
            datasets.push(new_series);
        }
    }

    var chartData = {
        labels: [],
        datasets: datasets
    };

    var chartOptions = {
        responsive: true,
        title: {
            display: true,
            text: 'Pool dynamics' //todo make configurable
        },
        interaction: {
          mode: 'nearest',
          axis: 'x',
          intersect: false
        },
        scales: {
            // note: using lists for the axes to accommodate stacking
            xAxes: [{
                //type: 'time',
                stacked: true,
                display: true,
                scaleLabel: {
                    display: true,
                    labelString: 'Iteration'
                },
                ticks: {
                  autoSkip: true,
                  maxTicksLimit: 11
                }
            }],
            yAxes: [{
                stacked: true,
                display: true,
                scaleLabel: {
                    display: true,
                    labelString: 'Pool size (stake)'
                },
            }]
        },
        tooltips: {
            // only show active pools (stake > 0) on hover todo maybe do sth similar for legend
            filter: function (tooltipItem) {
                return tooltipItem.yLabel != 0
            },
            mode: 'index',
            intersect: false
        },
        elements: {
            point:{
                radius: 0 //to hide points from line
            }
        },
        hover: {
            mode: 'point',
            intersect: true
        },
        legend: {
            display: false
        }

    };

    var chart = new Chart(context, {
        type: 'line',
        data: chartData,
        options: chartOptions
    });

    this.render = function(data) {
        chart.data.labels.push(control.tick);
        for (i = 0; i < data.length; i++) {
            chart.data.datasets[i].data.push(data[i]);
        }
        chart.update();
    };

    this.reset = function() {
        while (chart.data.labels.length) { chart.data.labels.pop(); }
        chart.data.datasets.forEach(function(dataset) {
            while (dataset.data.length) { dataset.data.pop(); }
        });
        datasets_number = series[0].Num_agents;
        console.log(datasets_number)

        chart.update();
    };
};